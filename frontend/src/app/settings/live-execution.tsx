import { type FormEvent, useEffect, useState } from "react";
import { Link } from "react-router";
import { useTranslation } from "react-i18next";
import {
  AlertTriangle,
  KeyRound,
  RefreshCw,
  ShieldAlert,
  ShieldCheck,
} from "lucide-react";
import {
  useConfirmStartupAuthorization,
  useCreateLiveConnection,
  useCreateLiveOrder,
  useCreateLiveStrategyBinding,
  useLiveConnections,
  useLiveExecutionStatus,
  useLiveRiskPolicy,
  useLiveStrategyBindings,
  useRequestStartupAuthorizationChallenge,
  useRevokeLiveStrategyBinding,
  useRevokeStartupAuthorization,
  useSaveLiveRiskPolicy,
} from "@/api/live-execution";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type {
  LiveExchangeProvider,
  LiveMarketType,
  LiveOrderSide,
  LiveOrderType,
} from "@/types/live-execution";

const numberOrZero = (value: string) => Number(value) || 0;

export default function LiveExecutionPage() {
  const { t } = useTranslation();
  const { data: status, isLoading: statusLoading } = useLiveExecutionStatus();
  const { data: connections = [] } = useLiveConnections();
  const { data: riskPolicy } = useLiveRiskPolicy();
  const { data: bindings = [] } = useLiveStrategyBindings();
  const createConnection = useCreateLiveConnection();
  const saveRiskPolicy = useSaveLiveRiskPolicy();
  const createBinding = useCreateLiveStrategyBinding();
  const revokeBinding = useRevokeLiveStrategyBinding();
  const requestChallenge = useRequestStartupAuthorizationChallenge();
  const confirmAuthorization = useConfirmStartupAuthorization();
  const revokeAuthorization = useRevokeStartupAuthorization();
  const createOrder = useCreateLiveOrder();

  const [provider, setProvider] = useState<LiveExchangeProvider>("binance");
  const [marketType, setMarketType] = useState<LiveMarketType>("spot");
  const [label, setLabel] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [apiSecret, setApiSecret] = useState("");
  const [passphrase, setPassphrase] = useState("");
  const [withdrawalDisabled, setWithdrawalDisabled] = useState(false);
  const [ipAllowlistConfigured, setIpAllowlistConfigured] = useState(false);
  const [maxNotional, setMaxNotional] = useState("");
  const [maxPositions, setMaxPositions] = useState("");
  const [maxLeverage, setMaxLeverage] = useState("");
  const [allowedSymbols, setAllowedSymbols] = useState("");
  const [strategyId, setStrategyId] = useState("");
  const [bindingConnectionId, setBindingConnectionId] = useState("");
  const [challengeCode, setChallengeCode] = useState("");
  const [confirmationCode, setConfirmationCode] = useState("");
  const [orderConnectionId, setOrderConnectionId] = useState("");
  const [symbol, setSymbol] = useState("BTC/USDT");
  const [side, setSide] = useState<LiveOrderSide>("buy");
  const [orderType, setOrderType] = useState<LiveOrderType>("market");
  const [quoteAmount, setQuoteAmount] = useState("");
  const [price, setPrice] = useState("");

  useEffect(() => {
    if (!riskPolicy) return;
    setMaxNotional(String(riskPolicy.max_order_notional));
    setMaxPositions(String(riskPolicy.max_open_positions));
    setMaxLeverage(String(riskPolicy.max_leverage));
    setAllowedSymbols(riskPolicy.allowed_symbols.join(", "));
  }, [riskPolicy]);

  const handleConnection = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (
      !label ||
      !apiKey ||
      !apiSecret ||
      !withdrawalDisabled ||
      !ipAllowlistConfigured ||
      (provider === "okx" && !passphrase)
    )
      return;
    try {
      await createConnection.mutateAsync({
        label,
        provider,
        market_type: marketType,
        api_key: apiKey,
        api_secret: apiSecret,
        ...(provider === "okx" ? { passphrase } : {}),
        withdrawal_disabled_confirmed: withdrawalDisabled,
        ip_allowlist_confirmed: ipAllowlistConfigured,
      });
      setLabel("");
      setApiKey("");
      setApiSecret("");
      setPassphrase("");
      setWithdrawalDisabled(false);
      setIpAllowlistConfigured(false);
    } finally {
      setApiKey("");
      setApiSecret("");
      setPassphrase("");
    }
  };

  const handleRiskPolicy = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    await saveRiskPolicy.mutateAsync({
      max_order_notional: numberOrZero(maxNotional),
      max_open_positions: numberOrZero(maxPositions),
      max_leverage: numberOrZero(maxLeverage),
      allowed_symbols: allowedSymbols
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean),
    });
  };

  const handleBinding = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!strategyId || !bindingConnectionId) return;
    await createBinding.mutateAsync({
      strategy_id: strategyId,
      connection_id: bindingConnectionId,
    });
    setStrategyId("");
  };

  const handleChallenge = async () => {
    const response = await requestChallenge.mutateAsync();
    setChallengeCode(response.data.challenge_code);
    setConfirmationCode("");
  };

  const handleAuthorization = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!confirmationCode) return;
    await confirmAuthorization.mutateAsync({
      challenge_code: confirmationCode,
    });
    setChallengeCode("");
    setConfirmationCode("");
  };

  const handleOrder = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const amount = Number(quoteAmount);
    const limitPrice = Number(price);
    if (
      !orderConnectionId ||
      !Number.isFinite(amount) ||
      amount <= 0 ||
      (orderType === "limit" &&
        (!Number.isFinite(limitPrice) || limitPrice <= 0))
    )
      return;
    await createOrder.mutateAsync({
      connection_id: orderConnectionId,
      symbol,
      side,
      type: orderType,
      quote_amount: amount,
      ...(orderType === "limit" ? { price: limitPrice } : {}),
      idempotency_key: crypto.randomUUID(),
    });
    setQuoteAmount("");
    setPrice("");
  };

  const orderEnabled = Boolean(
    status?.live_trading_enabled && status.authorization_active,
  );
  const gateReasons = status?.gate_reasons ?? [];

  return (
    <div className="scroll-container flex flex-1 flex-col gap-8 p-6 sm:p-10">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="font-bold text-xl">
            {t("settings.liveExecution.title")}
          </h1>
          <p className="mt-1 text-muted-foreground text-sm">
            {t("settings.liveExecution.description")}
          </p>
        </div>
        <Button variant="outline" asChild>
          <Link to="/settings">{t("settings.liveExecution.back")}</Link>
        </Button>
      </header>
      <Alert variant={status?.live_trading_enabled ? "default" : "destructive"}>
        <ShieldAlert />
        <AlertTitle>
          {statusLoading
            ? t("settings.liveExecution.status.loading")
            : status?.live_trading_enabled
              ? t("settings.liveExecution.status.enabled")
              : t("settings.liveExecution.status.disabled")}
        </AlertTitle>
        <AlertDescription>
          {t("settings.liveExecution.status.description")}
        </AlertDescription>
      </Alert>
      {gateReasons.length > 0 && (
        <Alert>
          <AlertTriangle />
          <AlertTitle>{t("settings.liveExecution.gates.title")}</AlertTitle>
          <AlertDescription>
            <ul className="list-disc space-y-1 pl-5">
              {gateReasons.map((reason) => (
                <li key={reason}>{reason}</li>
              ))}
            </ul>
          </AlertDescription>
        </Alert>
      )}

      <section className="grid gap-8 xl:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>
              {t("settings.liveExecution.connection.title")}
            </CardTitle>
            <CardDescription>
              {t("settings.liveExecution.connection.description")}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form className="space-y-4" onSubmit={handleConnection}>
              <div className="grid gap-4 sm:grid-cols-2">
                <div className="space-y-2">
                  <Label htmlFor="live-label">
                    {t("settings.liveExecution.connection.label")}
                  </Label>
                  <Input
                    id="live-label"
                    value={label}
                    onChange={(event) => setLabel(event.target.value)}
                    required
                  />
                </div>
                <div className="space-y-2">
                  <Label>
                    {t("settings.liveExecution.connection.provider")}
                  </Label>
                  <Select
                    value={provider}
                    onValueChange={(value) =>
                      setProvider(value as LiveExchangeProvider)
                    }
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="binance">Binance</SelectItem>
                      <SelectItem value="okx">OKX</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <div className="space-y-2">
                <Label>
                  {t("settings.liveExecution.connection.marketType")}
                </Label>
                <Select
                  value={marketType}
                  onValueChange={(value) =>
                    setMarketType(value as LiveMarketType)
                  }
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="spot">
                      {t("settings.liveExecution.marketTypes.spot")}
                    </SelectItem>
                    <SelectItem value="swap">
                      {t("settings.liveExecution.marketTypes.swap")}
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="live-key">
                  {t("settings.liveExecution.connection.apiKey")}
                </Label>
                <Input
                  id="live-key"
                  autoComplete="off"
                  value={apiKey}
                  onChange={(event) => setApiKey(event.target.value)}
                  required
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="live-secret">
                  {t("settings.liveExecution.connection.apiSecret")}
                </Label>
                <Input
                  id="live-secret"
                  type="password"
                  autoComplete="new-password"
                  value={apiSecret}
                  onChange={(event) => setApiSecret(event.target.value)}
                  required
                />
              </div>
              {provider === "okx" && (
                <div className="space-y-2">
                  <Label htmlFor="live-passphrase">
                    {t("settings.liveExecution.connection.passphrase")}
                  </Label>
                  <Input
                    id="live-passphrase"
                    type="password"
                    autoComplete="new-password"
                    value={passphrase}
                    onChange={(event) => setPassphrase(event.target.value)}
                    required
                  />
                </div>
              )}
              <div className="space-y-3">
                <div className="flex items-start gap-2">
                  <Checkbox
                    id="withdrawal-disabled"
                    checked={withdrawalDisabled}
                    onCheckedChange={(checked) =>
                      setWithdrawalDisabled(checked === true)
                    }
                  />
                  <Label htmlFor="withdrawal-disabled" className="leading-5">
                    {t("settings.liveExecution.connection.withdrawalDisabled")}
                  </Label>
                </div>
                <div className="flex items-start gap-2">
                  <Checkbox
                    id="ip-allowlist"
                    checked={ipAllowlistConfigured}
                    onCheckedChange={(checked) =>
                      setIpAllowlistConfigured(checked === true)
                    }
                  />
                  <Label htmlFor="ip-allowlist" className="leading-5">
                    {t("settings.liveExecution.connection.ipAllowlist")}
                  </Label>
                </div>
              </div>
              <Button
                type="submit"
                disabled={
                  createConnection.isPending ||
                  !withdrawalDisabled ||
                  !ipAllowlistConfigured
                }
              >
                {createConnection.isPending && (
                  <RefreshCw className="animate-spin" />
                )}
                {t("settings.liveExecution.connection.submit")}
              </Button>
            </form>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>{t("settings.liveExecution.risk.title")}</CardTitle>
            <CardDescription>
              {t("settings.liveExecution.risk.description")}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form className="space-y-4" onSubmit={handleRiskPolicy}>
              <div className="grid gap-4 sm:grid-cols-3">
                <div className="space-y-2">
                  <Label htmlFor="max-notional">
                    {t("settings.liveExecution.risk.maxNotional")}
                  </Label>
                  <Input
                    id="max-notional"
                    type="number"
                    min="0"
                    value={maxNotional}
                    onChange={(event) => setMaxNotional(event.target.value)}
                    required
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="max-positions">
                    {t("settings.liveExecution.risk.maxPositions")}
                  </Label>
                  <Input
                    id="max-positions"
                    type="number"
                    min="0"
                    step="1"
                    value={maxPositions}
                    onChange={(event) => setMaxPositions(event.target.value)}
                    required
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="max-leverage">
                    {t("settings.liveExecution.risk.maxLeverage")}
                  </Label>
                  <Input
                    id="max-leverage"
                    type="number"
                    min="0"
                    value={maxLeverage}
                    onChange={(event) => setMaxLeverage(event.target.value)}
                    required
                  />
                </div>
              </div>
              <div className="space-y-2">
                <Label htmlFor="allowed-symbols">
                  {t("settings.liveExecution.risk.allowedSymbols")}
                </Label>
                <Input
                  id="allowed-symbols"
                  value={allowedSymbols}
                  onChange={(event) => setAllowedSymbols(event.target.value)}
                  placeholder="BTC/USDT, ETH/USDT"
                  required
                />
              </div>
              <Button type="submit" disabled={saveRiskPolicy.isPending}>
                {t("settings.liveExecution.risk.submit")}
              </Button>
            </form>
          </CardContent>
        </Card>
      </section>

      <section className="grid gap-8 xl:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>{t("settings.liveExecution.binding.title")}</CardTitle>
            <CardDescription>
              {t("settings.liveExecution.binding.description")}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-5">
            <form className="space-y-4" onSubmit={handleBinding}>
              <div className="space-y-2">
                <Label htmlFor="strategy-id">
                  {t("settings.liveExecution.binding.strategyId")}
                </Label>
                <Input
                  id="strategy-id"
                  value={strategyId}
                  onChange={(event) => setStrategyId(event.target.value)}
                  required
                />
              </div>
              <div className="space-y-2">
                <Label>{t("settings.liveExecution.binding.connection")}</Label>
                <Select
                  value={bindingConnectionId}
                  onValueChange={setBindingConnectionId}
                >
                  <SelectTrigger>
                    <SelectValue
                      placeholder={t(
                        "settings.liveExecution.binding.connectionPlaceholder",
                      )}
                    />
                  </SelectTrigger>
                  <SelectContent>
                    {connections
                      .filter((connection) => connection.active)
                      .map((connection) => (
                        <SelectItem key={connection.id} value={connection.id}>
                          {connection.label} ({connection.provider}/
                          {connection.market_type})
                        </SelectItem>
                      ))}
                  </SelectContent>
                </Select>
              </div>
              <Button type="submit" disabled={createBinding.isPending}>
                {t("settings.liveExecution.binding.submit")}
              </Button>
            </form>
            <div className="space-y-2 border-t pt-4">
              {bindings.length === 0 ? (
                <p className="text-muted-foreground text-sm">
                  {t("settings.liveExecution.binding.empty")}
                </p>
              ) : (
                bindings.map((binding) => (
                  <div
                    className="flex flex-wrap items-center justify-between gap-3 text-sm"
                    key={binding.id}
                  >
                    <span>{binding.strategy_id}</span>
                    <div className="flex items-center gap-2">
                      <Badge
                        variant={
                          binding.active && !binding.revoked_at
                            ? "default"
                            : "secondary"
                        }
                      >
                        {binding.active && !binding.revoked_at
                          ? t("settings.liveExecution.binding.active")
                          : t("settings.liveExecution.binding.revoked")}
                      </Badge>
                      {binding.active && !binding.revoked_at && (
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => revokeBinding.mutate(binding.id)}
                          disabled={revokeBinding.isPending}
                        >
                          {t("settings.liveExecution.binding.revoke")}
                        </Button>
                      )}
                    </div>
                  </div>
                ))
              )}
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>
              {t("settings.liveExecution.authorization.title")}
            </CardTitle>
            <CardDescription>
              {t("settings.liveExecution.authorization.description")}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {status?.authorization_active ? (
              <div className="space-y-3">
                <div className="flex items-center gap-2 text-sm">
                  <ShieldCheck className="text-green-600" />
                  {t("settings.liveExecution.authorization.active", {
                    expiresAt: status.authorization_expires_at ?? "",
                  })}
                </div>
                <Button
                  variant="destructive"
                  onClick={() => revokeAuthorization.mutate()}
                  disabled={revokeAuthorization.isPending}
                >
                  {t("settings.liveExecution.authorization.revoke")}
                </Button>
              </div>
            ) : (
              <>
                <Button
                  type="button"
                  variant="outline"
                  onClick={handleChallenge}
                  disabled={requestChallenge.isPending}
                >
                  {requestChallenge.isPending && (
                    <RefreshCw className="animate-spin" />
                  )}
                  {t("settings.liveExecution.authorization.request")}
                </Button>
                {challengeCode && (
                  <form
                    className="space-y-4 border-t pt-4"
                    onSubmit={handleAuthorization}
                  >
                    <div className="space-y-2">
                      <Label htmlFor="challenge-display">
                        {t("settings.liveExecution.authorization.challenge")}
                      </Label>
                      <Input
                        id="challenge-display"
                        value={challengeCode}
                        readOnly
                        aria-describedby="challenge-help"
                      />
                      <p
                        id="challenge-help"
                        className="text-muted-foreground text-sm"
                      >
                        {t("settings.liveExecution.authorization.manualEntry")}
                      </p>
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="challenge-confirmation">
                        {t("settings.liveExecution.authorization.confirmation")}
                      </Label>
                      <Input
                        id="challenge-confirmation"
                        autoComplete="one-time-code"
                        value={confirmationCode}
                        onChange={(event) =>
                          setConfirmationCode(event.target.value)
                        }
                        required
                      />
                    </div>
                    <Button
                      type="submit"
                      disabled={confirmAuthorization.isPending}
                    >
                      {t("settings.liveExecution.authorization.confirm")}
                    </Button>
                  </form>
                )}
              </>
            )}
          </CardContent>
        </Card>
      </section>

      <Card className="border-destructive/50">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-destructive">
            <KeyRound />
            {t("settings.liveExecution.order.title")}
          </CardTitle>
          <CardDescription>
            {t("settings.liveExecution.order.description")}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form
            className="grid gap-4 md:grid-cols-2 xl:grid-cols-3"
            onSubmit={handleOrder}
          >
            <div className="space-y-2">
              <Label>{t("settings.liveExecution.order.connection")}</Label>
              <Select
                value={orderConnectionId}
                onValueChange={setOrderConnectionId}
                disabled={!orderEnabled}
              >
                <SelectTrigger>
                  <SelectValue
                    placeholder={t(
                      "settings.liveExecution.order.connectionPlaceholder",
                    )}
                  />
                </SelectTrigger>
                <SelectContent>
                  {connections
                    .filter((connection) => connection.active)
                    .map((connection) => (
                      <SelectItem key={connection.id} value={connection.id}>
                        {connection.label}
                      </SelectItem>
                    ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="order-symbol">
                {t("settings.liveExecution.order.symbol")}
              </Label>
              <Input
                id="order-symbol"
                value={symbol}
                onChange={(event) => setSymbol(event.target.value)}
                disabled={!orderEnabled}
                required
              />
            </div>
            <div className="space-y-2">
              <Label>{t("settings.liveExecution.order.side")}</Label>
              <Select
                value={side}
                onValueChange={(value) => setSide(value as LiveOrderSide)}
                disabled={!orderEnabled}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="buy">
                    {t("settings.liveExecution.order.sides.buy")}
                  </SelectItem>
                  <SelectItem value="sell">
                    {t("settings.liveExecution.order.sides.sell")}
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>{t("settings.liveExecution.order.type")}</Label>
              <Select
                value={orderType}
                onValueChange={(value) => setOrderType(value as LiveOrderType)}
                disabled={!orderEnabled}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="market">
                    {t("settings.liveExecution.order.types.market")}
                  </SelectItem>
                  <SelectItem value="limit">
                    {t("settings.liveExecution.order.types.limit")}
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="quote-amount">
                {t("settings.liveExecution.order.quoteAmount")}
              </Label>
              <Input
                id="quote-amount"
                type="number"
                min="0"
                value={quoteAmount}
                onChange={(event) => setQuoteAmount(event.target.value)}
                disabled={!orderEnabled}
                required
              />
            </div>
            {orderType === "limit" && (
              <div className="space-y-2">
                <Label htmlFor="limit-price">
                  {t("settings.liveExecution.order.price")}
                </Label>
                <Input
                  id="limit-price"
                  type="number"
                  min="0"
                  value={price}
                  onChange={(event) => setPrice(event.target.value)}
                  disabled={!orderEnabled}
                  required
                />
              </div>
            )}
            <div className="flex items-end">
              <Button
                variant="destructive"
                type="submit"
                disabled={!orderEnabled || createOrder.isPending}
              >
                {t("settings.liveExecution.order.submit")}
              </Button>
            </div>
          </form>
          {!orderEnabled && (
            <p className="mt-4 text-muted-foreground text-sm">
              {t("settings.liveExecution.order.unavailable")}
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
