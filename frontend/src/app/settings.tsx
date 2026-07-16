import { Link } from "react-router";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import GeneralPage from "./setting/general";

export default function SettingsPage() {
  const { t } = useTranslation();
  return (
    <div className="flex flex-1 flex-col">
      <div className="flex flex-wrap justify-end gap-3 px-10 pt-6">
        <Button variant="outline" asChild>
          <Link to="/settings/sandbox-exchanges">
            {t("settings.sandboxExchanges.link")}
          </Link>
        </Button>
        <Button
          variant="outline"
          className="border-destructive/50 text-destructive hover:text-destructive"
          asChild
        >
          <Link to="/settings/live-execution">
            {t("settings.liveExecution.link")}
          </Link>
        </Button>
      </div>
      <GeneralPage />
    </div>
  );
}
