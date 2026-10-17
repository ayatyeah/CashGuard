import { FileVideo, KeyRound, LayoutDashboard, ShieldCheck } from "lucide-react";

import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Sidebar, SidebarMenuButton, useSidebar } from "@/components/ui/app-1-utils/sidebar";
import { useWorkspace } from "@/lib/workspace";

export type Page = "dashboard" | "analysis";

export function App1Sidebar({
  page,
  onNavigate,
  activeCount,
  onConnect,
}: {
  page: Page;
  onNavigate: (page: Page) => void;
  activeCount: number;
  onConnect: () => void;
}) {
  const { t, token } = useWorkspace();
  const { isMobile, setOpenMobile } = useSidebar();
  const go = (next: Page) => {
    onNavigate(next);
    if (isMobile) setOpenMobile(false);
  };

  return (
    <Sidebar title={t("Рабочее пространство")}>
      <button type="button" onClick={() => go("dashboard")} className="cashguard-brand" aria-label={t("Обзор проверок")}>
        <img src="/brand/cashguard-forte.svg" alt="" width="38" height="38" />
        <span><strong>Cash<span>Guard</span></strong><small>FORTE · {t("Кассовый контроль")}</small></span>
      </button>

      <nav className="flex flex-col gap-1 p-3">
        <p className="px-3 pb-2 pt-1 text-xs font-medium text-muted-foreground">{t("Рабочее пространство")}</p>
        <SidebarMenuButton active={page === "dashboard"} onClick={() => go("dashboard")}>
          <LayoutDashboard />
          {t("Обзор")}
        </SidebarMenuButton>
        <SidebarMenuButton active={page === "analysis"} onClick={() => go("analysis")}>
          <FileVideo />
          {t("Анализ видео")}
          {activeCount > 0 && (
            <Badge variant="secondary" className="ml-auto px-2 tabular-nums">
              {activeCount}
            </Badge>
          )}
        </SidebarMenuButton>
      </nav>

      <div className="mt-auto flex flex-col gap-3 p-3">
        <div className="rounded-lg border bg-white/[0.03] p-4 text-xs">
          <p className="flex items-center gap-2 font-medium">
            <ShieldCheck className="size-4 text-primary" aria-hidden />
            {t("Локальный контур")}
          </p>
          <p className="mt-2 leading-relaxed text-muted-foreground">
            {t("Видео обрабатывается внутри вашей инфраструктуры.")}
          </p>
          <p className="mt-2 flex items-center gap-1.5 text-primary">
            <span className="size-1.5 rounded-full bg-primary" aria-hidden />
            {t("Без облачных AI-сервисов")}
          </p>
        </div>
        <button
          onClick={onConnect}
          className="flex items-center gap-3 rounded-md p-2 text-left transition-colors hover:bg-sidebar-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          <Avatar className="size-8">
            <AvatarFallback className="bg-sidebar-accent text-sidebar-accent-foreground">
              <KeyRound className="size-4" />
            </AvatarFallback>
          </Avatar>
          <span className="min-w-0">
            <span className="block truncate text-sm font-medium">{t("Настройки подключения")}</span>
            <span className="block truncate text-xs text-muted-foreground">
              {t(token ? "API-ключ подключён" : "Подключить API")}
            </span>
          </span>
        </button>
      </div>
    </Sidebar>
  );
}
