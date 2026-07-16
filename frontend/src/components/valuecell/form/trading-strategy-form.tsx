import { MultiSelect } from "@valuecell/multi-select";
import { Eye, Plus, Trash2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import {
  useCreateStrategyPrompt,
  useDeleteStrategyPrompt,
} from "@/api/strategy";
import NewPromptModal from "@/app/agent/components/strategy-items/modals/new-prompt-modal";
import ViewStrategyModal from "@/app/agent/components/strategy-items/modals/view-strategy-modal";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import {
  Field,
  FieldError,
  FieldGroup,
  FieldLabel,
} from "@/components/ui/field";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { TRADING_SYMBOLS } from "@/constants/agent";
import { withForm } from "@/hooks/use-form";
import type {
  Strategy,
  StrategyConfigField,
  StrategyConfigSchema,
  StrategyPrompt,
} from "@/types/strategy";
type TradingConfigFieldKey =
  | "symbols"
  | "initial_capital"
  | "decide_interval"
  | "max_leverage"
  | "max_positions"
  | "cap_factor"
  | "template_id";

export const TradingStrategyForm = withForm({
  defaultValues: {
    strategy_type: "" as Strategy["strategy_type"],
    strategy_name: "",
    initial_capital: 1000,
    max_leverage: 2,
    decide_interval: 60,
    symbols: TRADING_SYMBOLS,
    template_id: "",
    max_positions: 5,
    cap_factor: 1,
    strategy_params: {} as Record<string, unknown>,
  },
  props: {
    prompts: [] as StrategyPrompt[],
    tradingMode: "live" as "live" | "virtual",
    strategySchemas: [] as StrategyConfigSchema[],
  },
  render({ form, prompts, tradingMode, strategySchemas }) {
    const { t } = useTranslation();
    const { mutateAsync: createStrategyPrompt } = useCreateStrategyPrompt();
    const { mutate: deleteStrategyPrompt } = useDeleteStrategyPrompt();
    const [deletePromptId, setDeletePromptId] = useState<string | null>(null);
    const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);

    const selectedSchema = useMemo(
      () =>
        strategySchemas.find(
          (schema) => schema.strategy_type === form.state.values.strategy_type,
        ),
      [strategySchemas, form.state.values.strategy_type],
    );

    const tradingConfigFields = useMemo(
      () =>
        (selectedSchema?.fields ?? []).filter(
          (field) => field.persistence_target === "trading_config",
        ),
      [selectedSchema],
    );

    const dynamicFields = useMemo(
      () =>
        (selectedSchema?.fields ?? []).filter(
          (field) => field.persistence_target === "strategy_params",
        ),
      [selectedSchema],
    );

    useEffect(() => {
      if (!selectedSchema) return;
      const defaults = selectedSchema.defaults ?? {};
      for (const field of tradingConfigFields) {
        form.setFieldValue(
          field.key as TradingConfigFieldKey,
          (defaults[field.key] ?? field.default) as never,
        );
      }
      form.setFieldValue(
        "strategy_params",
        Object.fromEntries(
          dynamicFields.map((field) => [
            field.key,
            defaults[field.key] ?? field.default,
          ]),
        ),
      );
    }, [selectedSchema, dynamicFields, form, tradingConfigFields]);

    const updateStrategyParam = (key: string, value: unknown) => {
      form.setFieldValue("strategy_params", {
        ...(form.state.values.strategy_params ?? {}),
        [key]: value,
      });
    };
    const fieldValue = (field: StrategyConfigField) =>
      field.persistence_target === "trading_config"
        ? form.state.values[field.key as TradingConfigFieldKey]
        : (form.state.values.strategy_params ?? {})[field.key];

    const updateFieldValue = (field: StrategyConfigField, value: unknown) => {
      if (field.persistence_target === "trading_config") {
        form.setFieldValue(field.key as TradingConfigFieldKey, value as never);
        return;
      }
      updateStrategyParam(field.key, value);
    };

    const renderDynamicField = (field: StrategyConfigField) => {
      const value = fieldValue(field);
      if (field.field_type === "boolean") {
        return (
          <Field
            key={field.key}
            orientation="horizontal"
            className="items-center justify-between rounded-lg border p-3"
          >
            <div>
              <FieldLabel>{field.label}</FieldLabel>
              {field.description && (
                <p className="text-muted-foreground text-xs">
                  {field.description}
                </p>
              )}
            </div>
            <input
              type="checkbox"
              checked={Boolean(value)}
              onChange={(event) =>
                updateFieldValue(field, event.target.checked)
              }
            />
          </Field>
        );
      }
      if (field.field_type === "select") {
        return (
          <Field key={field.key}>
            <FieldLabel>{field.label}</FieldLabel>
            <Select
              value={String(value ?? field.default ?? "")}
              onValueChange={(nextValue) => updateFieldValue(field, nextValue)}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {field.options.map((option) => (
                  <SelectItem
                    key={String(option.value)}
                    value={String(option.value)}
                  >
                    {option.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Field>
        );
      }
      if (field.field_type === "number_list") {
        const listValue = Array.isArray(value) ? value.join(",") : "";
        return (
          <Field key={field.key}>
            <FieldLabel>{field.label}</FieldLabel>
            <input
              className="h-10 rounded-md border bg-background px-3 text-sm"
              value={listValue}
              onChange={(event) =>
                updateFieldValue(
                  field,
                  event.target.value
                    .split(",")
                    .map((item) => Number(item.trim()))
                    .filter((item) => Number.isFinite(item)),
                )
              }
            />
            {field.description && (
              <p className="text-muted-foreground text-xs">
                {field.description}
              </p>
            )}
          </Field>
        );
      }
      return (
        <Field key={field.key}>
          <FieldLabel>{field.label}</FieldLabel>
          <input
            className="h-10 rounded-md border bg-background px-3 text-sm"
            type={field.field_type === "number" ? "number" : "text"}
            min={field.min}
            max={field.max}
            step={field.step}
            value={String(value ?? field.default ?? "")}
            onChange={(event) =>
              updateFieldValue(
                field,
                field.field_type === "number"
                  ? Number(event.target.value)
                  : event.target.value,
              )
            }
          />
          {field.description && (
            <p className="text-muted-foreground text-xs">{field.description}</p>
          )}
        </Field>
      );
    };

    const handleDeletePrompt = (promptId: string) => {
      setDeletePromptId(promptId);
      setIsDeleteDialogOpen(true);
    };

    const confirmDeletePrompt = () => {
      if (deletePromptId) {
        deleteStrategyPrompt(deletePromptId, {
          onSuccess: () => {
            // If the deleted prompt was currently selected, clear the selection
            if (form.state.values.template_id === deletePromptId) {
              form.setFieldValue("template_id", "");
            }
            setIsDeleteDialogOpen(false);
            setDeletePromptId(null);
          },
          onError: () => {
            setIsDeleteDialogOpen(false);
            setDeletePromptId(null);
          },
        });
      }
    };

    const cancelDeletePrompt = () => {
      setIsDeleteDialogOpen(false);
      setDeletePromptId(null);
    };

    return (
      <FieldGroup className="gap-6">
        <form.AppField
          listeners={{
            onChange: ({ value }: { value: Strategy["strategy_type"] }) => {
              const schema = strategySchemas.find(
                (item) => item.strategy_type === value,
              );
              const defaults = schema?.defaults ?? {};
              for (const field of schema?.fields ?? []) {
                const defaultValue = defaults[field.key] ?? field.default;
                if (field.persistence_target === "trading_config") {
                  form.setFieldValue(
                    field.key as TradingConfigFieldKey,
                    defaultValue as never,
                  );
                }
              }
              form.setFieldValue(
                "strategy_params",
                Object.fromEntries(
                  (schema?.fields ?? [])
                    .filter(
                      (field) => field.persistence_target === "strategy_params",
                    )
                    .map((field) => [
                      field.key,
                      defaults[field.key] ?? field.default,
                    ]),
                ),
              );
            },
          }}
          name="strategy_type"
        >
          {(field) => (
            <field.SelectField label={t("strategy.form.strategyType.label")}>
              {(strategySchemas.length > 0
                ? strategySchemas
                : [
                    {
                      strategy_type: "PromptBasedStrategy",
                      label: t("strategy.form.strategyType.promptBased"),
                    },
                    {
                      strategy_type: "GridStrategy",
                      label: t("strategy.form.strategyType.grid"),
                    },
                  ]
              ).map((schema) => (
                <SelectItem
                  key={schema.strategy_type}
                  value={schema.strategy_type}
                >
                  {schema.label}
                </SelectItem>
              ))}
            </field.SelectField>
          )}
        </form.AppField>

        <form.AppField name="strategy_name">
          {(field) => (
            <field.TextField
              label={t("strategy.form.strategyName.label")}
              placeholder={t("strategy.form.strategyName.placeholder")}
            />
          )}
        </form.AppField>

        {tradingConfigFields
          .filter(
            (field) =>
              !["symbols", "template_id"].includes(field.key) &&
              !(tradingMode === "live" && field.key === "initial_capital"),
          )
          .map(renderDynamicField)}

        <form.Field name="symbols">
          {(field) => (
            <Field>
              <FieldLabel className="font-medium text-base text-foreground">
                {selectedSchema?.fields.find((item) => item.key === "symbols")
                  ?.label ?? t("strategy.form.tradingSymbols.label")}
              </FieldLabel>
              <MultiSelect
                maxSelected={
                  selectedSchema?.strategy_type === "GridStrategy"
                    ? 1
                    : undefined
                }
                options={
                  selectedSchema?.fields
                    .find((item) => item.key === "symbols")
                    ?.options.map((item) => String(item.value)) ??
                  TRADING_SYMBOLS
                }
                value={field.state.value}
                onValueChange={field.handleChange}
                placeholder={t("strategy.form.tradingSymbols.placeholder")}
                searchPlaceholder={t(
                  "strategy.form.tradingSymbols.searchPlaceholder",
                )}
                emptyText={t("strategy.form.tradingSymbols.emptyText")}
                maxDisplayed={5}
                creatable
              />
              <FieldError errors={field.state.meta.errors} />
            </Field>
          )}
        </form.Field>

        <form.Subscribe selector={(state) => state.values.strategy_type}>
          {(strategyType) => {
            return (
              strategyType === "PromptBasedStrategy" && (
                <form.Field name="template_id">
                  {(field) => (
                    <Field>
                      <FieldLabel className="font-medium text-base text-foreground">
                        {t("strategy.form.promptTemplate.label")}
                      </FieldLabel>
                      <div className="flex items-center gap-3">
                        <Select
                          value={field.state.value}
                          onValueChange={(value) => {
                            field.handleChange(value);
                          }}
                        >
                          <SelectTrigger className="flex-1">
                            <SelectValue />
                          </SelectTrigger>

                          <SelectContent>
                            {prompts.length > 0 &&
                              prompts.map((prompt) => (
                                <SelectItem
                                  key={prompt.id}
                                  value={prompt.id}
                                  className="relative hover:[&_button]:opacity-100 hover:[&_button]:transition-opacity"
                                >
                                  <span>{prompt.name}</span>
                                  {field.state.value !== prompt.id && (
                                    <button
                                      type="button"
                                      className="absolute right-2 z-50 flex size-3.5 items-center justify-center rounded-sm p-0 opacity-0 transition-all hover:bg-destructive/10 hover:text-destructive hover:opacity-100"
                                      onPointerUp={(e) => {
                                        e.stopPropagation();
                                        e.preventDefault();
                                        handleDeletePrompt(prompt.id);
                                      }}
                                    >
                                      <Trash2 className="h-3 w-3" />
                                    </button>
                                  )}
                                </SelectItem>
                              ))}
                            <NewPromptModal
                              onSave={async (value) => {
                                const { data: prompt } =
                                  await createStrategyPrompt(value);
                                form.setFieldValue("template_id", prompt.id);
                              }}
                            >
                              <Button
                                className="w-full"
                                type="button"
                                variant="outline"
                              >
                                <Plus />
                                {t("strategy.form.promptTemplate.new")}
                              </Button>
                            </NewPromptModal>
                          </SelectContent>
                        </Select>

                        <ViewStrategyModal
                          prompt={prompts.find(
                            (prompt) => prompt.id === field.state.value,
                          )}
                        >
                          <Button type="button" variant="outline">
                            <Eye />
                            {t("strategy.form.promptTemplate.view")}
                          </Button>
                        </ViewStrategyModal>
                      </div>
                      <FieldError errors={field.state.meta.errors} />
                    </Field>
                  )}
                </form.Field>
              )
            );
          }}
        </form.Subscribe>

        {dynamicFields.length > 0 && (
          <FieldGroup className="rounded-lg border bg-muted/20 p-4">
            <div>
              <FieldLabel className="text-base">策略动态参数</FieldLabel>
              <p className="text-muted-foreground text-sm">
                {selectedSchema?.description}
              </p>
            </div>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              {dynamicFields.map(renderDynamicField)}
            </div>
          </FieldGroup>
        )}

        {/* Delete Confirmation Dialog */}
        <AlertDialog
          open={isDeleteDialogOpen}
          onOpenChange={setIsDeleteDialogOpen}
        >
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>
                {t("strategy.prompt.delete.title")}
              </AlertDialogTitle>
              <AlertDialogDescription>
                {t("strategy.prompt.delete.description")}
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel onClick={cancelDeletePrompt}>
                {t("strategy.action.cancel")}
              </AlertDialogCancel>
              <AlertDialogAction
                onClick={confirmDeletePrompt}
                className="bg-destructive text-white hover:bg-destructive/90 focus-visible:ring-destructive/20"
              >
                {t("strategy.action.confirmDelete")}
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </FieldGroup>
    );
  },
});
