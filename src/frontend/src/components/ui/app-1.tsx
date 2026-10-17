import { useEffect, useRef, useState, type ReactNode } from "react";
import { Area, AreaChart, CartesianGrid, XAxis, YAxis } from "recharts";
import {
  Activity,
  ArrowDownToLine,
  ArrowRight,
  Clock3,
  FileVideo,
  KeyRound,
  Loader2,
  LockKeyhole,
  Play,
  Plus,
  Search,
  ShieldAlert,
  ShieldCheck,
  UploadCloud,
  Video,
  X,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { SidebarInset, SidebarProvider, SidebarTrigger } from "@/components/ui/app-1-utils/sidebar";
import { App1Sidebar, type Page } from "@/components/ui/app-1-utils/app-1-sidebar";
import { AccountDialog } from "./account-dialog";
import { Preferences } from "./preferences";
import { StoredVideo } from "./stored-video";
import { api, ApiError, timecode, type Job } from "@/lib/api";
import { formatDate } from "@/lib/translations";
import { cn } from "@/lib/utils";
import { useWorkspace } from "@/lib/workspace";

type Filter = "all" | "flagged" | "active" | "clean";

const isActive = (j: Job) => j.status === "queued" || j.status === "processing";

// One colour per verdict across the whole UI: amber = needs review, green = clean, red = failed.
function statusTone(j: Job) {
  if (j.status === "failed") return "border-transparent bg-destructive-soft text-destructive";
  if (isActive(j)) return "border-transparent bg-secondary text-secondary-foreground";
  return j.verdict === "violation"
    ? "border-transparent bg-warning-soft text-warning-foreground"
    : "border-transparent bg-success-soft text-success-foreground";
}

function Notice({ tone, icon, children, action }: { tone: "info" | "warning" | "error"; icon: ReactNode; children: ReactNode; action?: ReactNode }) {
  return (
    <div
      role={tone === "error" ? "alert" : undefined}
      className={cn(
        "flex flex-wrap items-center gap-3 rounded-lg border px-4 py-3 text-sm sm:flex-nowrap [&>svg]:size-5 [&>svg]:shrink-0",
        tone === "info" && "bg-white/[0.04] text-foreground backdrop-blur-xl",
        tone === "warning" && "border-warning-soft bg-warning-soft text-warning-foreground",
        tone === "error" && "border-destructive-soft bg-destructive-soft text-destructive",
      )}
    >
      {icon}
      <div className="min-w-0 flex-1">{children}</div>
      {action}
    </div>
  );
}

export function App1() {
  const { token: key, t, locale, language, settingsError, invalidate } = useWorkspace();
  const [page, setPage] = useState<Page>("dashboard");
  const [jobs, setJobs] = useState<Job[]>([]);
  const [keyDialog, setKeyDialog] = useState(false);
  const [error, setError] = useState("");
  const [connectionError, setConnectionError] = useState("");
  const [connected, setConnected] = useState(false);
  const [demo, setDemo] = useState(false);
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<Filter>("all");
  const [selected, setSelected] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const [uploading, setUploading] = useState(false);
  const [drag, setDrag] = useState(false);
  const [localVideos, setLocalVideos] = useState<Record<string, string>>({});
  const videoRef = useRef<HTMLVideoElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const statusLabel = (j: Job) =>
    j.status === "failed"
      ? t("Ошибка обработки")
      : j.status === "queued"
        ? t("В очереди")
        : j.status === "processing"
          ? t("Анализируется")
          : j.verdict === "violation"
            ? t("Требует проверки")
            : t("Без нарушений");

  useEffect(() => {
    setJobs([]);
    setSelected(null);
    setFile(null);
    setConnected(false);
    setError("");
    setLocalVideos((old) => {
      Object.values(old).forEach(URL.revokeObjectURL);
      return {};
    });
    let cancelled = false;
    async function refresh() {
      try {
        const health = await api<{ demo: boolean }>("/health", "");
        if (!cancelled) {
          setDemo(health.demo);
          setConnected(true);
          setConnectionError("");
        }
        if (!key) return;
        const data = await api<Job[]>("/jobs", key);
        if (!cancelled) setJobs(data);
      } catch (e) {
        if (cancelled) return;
        setConnected(false);
        setConnectionError((e as Error).message);
        if (e instanceof ApiError && e.status === 401) invalidate();
      }
    }
    void refresh();
    const id = setInterval(refresh, 3000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [key]);

  useEffect(() => {
    if (!file) {
      setPreview("");
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const done = jobs.filter((j) => j.status === "done");
  const flagged = done.filter((j) => j.verdict === "violation");
  const clean = done.filter((j) => j.verdict === "no_violation");
  const failed = jobs.filter((j) => j.status === "failed");
  const active = jobs.filter(isActive);
  const current = jobs.find((j) => j.job_id === selected);
  const percent = (n: number) => (jobs.length ? Math.round((n / jobs.length) * 100) : 0);

  const chart = Array.from({ length: 7 }, (_, i) => {
    const date = new Date();
    date.setDate(date.getDate() - 6 + i);
    const day = done.filter((j) => new Date(j.created_at).toDateString() === date.toDateString());
    return { day: formatDate(date, language), checked: day.length, flagged: day.filter((j) => j.verdict === "violation").length };
  });
  const chartConfig = {
    checked: { label: t("Проверено видео"), color: "hsl(var(--primary))" },
    flagged: { label: t("Требуют проверки"), color: "hsl(var(--muted-foreground))" },
  } satisfies ChartConfig;

  const displayed = jobs.filter(
    (j) =>
      j.filename.toLowerCase().includes(search.toLowerCase()) &&
      (filter === "all" ||
        (filter === "flagged" && j.verdict === "violation") ||
        (filter === "active" && isActive(j)) ||
        (filter === "clean" && j.verdict === "no_violation")),
  );

  function choose(f?: File) {
    if (!f) return;
    if (!/\.(mp4|mov|webm)$/i.test(f.name) || f.size === 0 || f.size > 500 * 1024 * 1024) {
      setError(t("Выберите MP4, MOV или WebM размером до 500 МБ."));
      return;
    }
    setFile(f);
    setError("");
    setSelected(null);
  }

  async function upload() {
    if (!file) return;
    if (!key) {
      setKeyDialog(true);
      return;
    }
    setUploading(true);
    setError("");
    try {
      const body = new FormData();
      body.append("video", file);
      const j = await api<Job>("/jobs", key, { method: "POST", body });
      setJobs((old) => [j, ...old.filter((x) => x.job_id !== j.job_id)]);
      setLocalVideos((old) => ({ ...old, [j.job_id]: URL.createObjectURL(file) }));
      setSelected(j.job_id);
      setFile(null);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setUploading(false);
    }
  }

  function report(j: Job) {
    const url = URL.createObjectURL(new Blob([JSON.stringify(j, null, 2)], { type: "application/json" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `cashguard-${j.job_id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function open(j: Job) {
    setPage("analysis");
    setSelected(j.job_id);
    setFile(null);
  }

  function startNew() {
    setPage("analysis");
    setSelected(null);
  }

  const stats = [
    { label: t("Проверено видео"), value: done.length, hint: t("Завершённые проверки"), icon: Video },
    { label: t("Требуют проверки"), value: flagged.length, hint: t("Возможные нарушения"), icon: ShieldAlert },
    { label: t("Без нарушений"), value: clean.length, hint: t("Признаки не обнаружены"), icon: ShieldCheck },
    { label: t("В обработке"), value: active.length, hint: t("В очереди и на анализе"), icon: Clock3 },
  ];

  const breakdown = [
    { label: t("Без нарушений"), value: clean.length, bar: "[&>div]:bg-primary" },
    { label: t("Требуют проверки"), value: flagged.length, bar: "[&>div]:bg-warning" },
    { label: t("Ошибка обработки"), value: failed.length, bar: "[&>div]:bg-destructive" },
    { label: t("В обработке"), value: active.length, bar: "[&>div]:bg-muted-foreground" },
  ];

  const filters: [Filter, string][] = [
    ["all", t("Все видео")],
    ["flagged", t("Требуют проверки")],
    ["active", t("В обработке")],
    ["clean", t("Без нарушений")],
  ];

  return (
    <SidebarProvider>
      <App1Sidebar page={page} onNavigate={setPage} activeCount={active.length} onConnect={() => setKeyDialog(true)} />
      {/* min-w-0: lets the inset shrink below its widest child (the history table) next to the sidebar */}
      <SidebarInset className="min-w-0">
        <header className="sticky top-0 z-10 flex h-16 items-center gap-3 border-b bg-background/40 px-4 backdrop-blur-xl sm:px-6">
          <SidebarTrigger aria-label={t("Переключить меню")} />
          <h1 className="truncate text-lg font-semibold">{page === "dashboard" ? t("Обзор проверок") : t("Анализ видео")}</h1>
          <div className="ml-auto flex items-center gap-1 sm:gap-2">
            <span className="mr-1 hidden items-center gap-2 text-xs text-muted-foreground lg:flex">
              <span className={cn("size-2 rounded-full", connected ? "bg-primary" : "bg-muted-foreground/50")} aria-hidden />
              {connected ? t("Сервис подключён") : t("Нет подключения")}
            </span>
            <Preferences />
            <Button variant="ghost" size="icon" className="size-9" aria-label={t("Настройки подключения")} onClick={() => setKeyDialog(true)}>
              <KeyRound className="size-5" />
            </Button>
          </div>
        </header>

        <main className="mx-auto flex w-full max-w-[1400px] flex-1 flex-col gap-6 p-4 sm:p-6">
          <div className="flex flex-wrap items-end justify-between gap-4">
            <div>
              <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">{t("Видеоаналитика · безопасность")}</p>
              <p className="mt-1 text-sm text-muted-foreground">
                {page === "dashboard" ? t("Всё, что важно для контроля кассовых операций.") : t("Загрузите запись и проверьте наблюдаемые события.")}
              </p>
            </div>
            {page === "dashboard" && (
              <Button onClick={startNew} className="gap-2">
                <Plus className="size-4" />
                {t("Новый анализ")}
              </Button>
            )}
          </div>

          {demo && (
            <Notice tone="warning" icon={<Activity />}>
              <strong>{t("Демонстрационный режим.")}</strong> {t("Результаты синтетические; загруженное видео не анализируется.")}
            </Notice>
          )}
          {!key && (
            <Notice
              tone="info"
              icon={<LockKeyhole />}
              action={
                <Button variant="outline" size="sm" className="gap-2" onClick={() => setKeyDialog(true)}>
                  {t("Подключить")}
                  <ArrowRight className="size-4" />
                </Button>
              }
            >
              <p className="font-medium">{t("Общее рабочее пространство")}</p>
              <p className="opacity-80">{t("Подключите API-ключ, чтобы загружать видео и просматривать результаты.")}</p>
            </Notice>
          )}
          {settingsError && <Notice tone="error" icon={<X />}>{t(settingsError)}</Notice>}
          {(error || connectionError) && (
            <Notice
              tone="error"
              icon={<ShieldAlert />}
              action={
                <Button variant="ghost" size="icon" className="size-8" aria-label={t("Скрыть ошибку")} onClick={() => { setError(""); setConnectionError(""); }}>
                  <X className="size-4" />
                </Button>
              }
            >
              {t(error || connectionError)}
            </Notice>
          )}

          {page === "dashboard" ? (
            <>
              <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
                {stats.map((stat) => (
                  <Card key={stat.label}>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                      <CardTitle className="text-sm font-medium">{stat.label}</CardTitle>
                      <stat.icon aria-hidden className="size-5 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                      <p className="text-2xl font-semibold tabular-nums">{stat.value.toLocaleString(locale)}</p>
                      <p className="text-xs text-muted-foreground">{stat.hint}</p>
                    </CardContent>
                  </Card>
                ))}
              </div>

              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">{t("Активность проверок")}</CardTitle>
                  <CardDescription>{t("Количество обработанных видео за неделю")}</CardDescription>
                </CardHeader>
                <CardContent>
                  <ChartContainer config={chartConfig} className="aspect-auto h-64 w-full">
                    <AreaChart data={chart} margin={{ left: 4, right: 4, top: 8 }} accessibilityLayer>
                      <defs>
                        <linearGradient id="app1Checked" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="var(--color-checked)" stopOpacity={0.25} />
                          <stop offset="100%" stopColor="var(--color-checked)" stopOpacity={0.02} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid vertical={false} strokeDasharray="3 3" />
                      <XAxis dataKey="day" tickLine={false} axisLine={false} tickMargin={8} />
                      <YAxis allowDecimals={false} tickLine={false} axisLine={false} width={32} />
                      <ChartTooltip content={<ChartTooltipContent />} />
                      <ChartLegend content={<ChartLegendContent />} />
                      {/* the two series differ by form (dashed vs filled), not only by colour */}
                      <Area dataKey="flagged" type="monotone" stroke="var(--color-flagged)" strokeDasharray="4 4" fill="none" strokeWidth={2} />
                      <Area dataKey="checked" type="monotone" stroke="var(--color-checked)" fill="url(#app1Checked)" strokeWidth={2} />
                    </AreaChart>
                  </ChartContainer>
                </CardContent>
              </Card>

              <div className="grid gap-6 lg:grid-cols-2">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">{t("Контроль в деталях")}</CardTitle>
                    <CardDescription>{t("Распределение всех проверок по результату")}</CardDescription>
                  </CardHeader>
                  <CardContent className="flex flex-col gap-5">
                    {breakdown.map((row) => (
                      <div key={row.label} className="flex flex-col gap-2">
                        <div className="flex items-center justify-between gap-2 text-sm">
                          <span className="font-medium">{row.label}</span>
                          <span className="tabular-nums text-muted-foreground">
                            {row.value} · {percent(row.value)}%
                          </span>
                        </div>
                        <Progress value={percent(row.value)} aria-label={row.label} className={cn("h-2", row.bar)} />
                      </div>
                    ))}
                    <p className="border-t pt-4 text-xs leading-relaxed text-muted-foreground">
                      {t("Выводы сервиса помогают проверяющему принять решение.")}
                    </p>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="flex flex-row items-start justify-between gap-4 space-y-0">
                    <div className="space-y-1.5">
                      <CardTitle className="text-lg">{t("Последние проверки")}</CardTitle>
                      <CardDescription>{t("Записи и результаты в одном месте")}</CardDescription>
                    </div>
                    <Button variant="link" size="sm" className="h-auto gap-1 px-0" onClick={() => setPage("analysis")}>
                      {t("Все проверки")}
                      <ArrowRight className="size-4" />
                    </Button>
                  </CardHeader>
                  <CardContent>
                    {jobs.length === 0 ? (
                      <EmptyState
                        title={t("Здесь появятся ваши проверки")}
                        text={t("Загрузите первое видео, чтобы начать контроль кассовых операций.")}
                        action={<Button variant="outline" size="sm" onClick={startNew}>{t("Загрузить видео")}</Button>}
                      />
                    ) : (
                      <ol className="flex flex-col gap-1">
                        {jobs.slice(0, 6).map((j) => (
                          <li key={j.job_id}>
                            <button
                              onClick={() => open(j)}
                              className="flex w-full items-center gap-3 rounded-md p-2 text-left transition-colors hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                            >
                              <span className="grid size-9 shrink-0 place-items-center rounded-full bg-muted text-muted-foreground">
                                <FileVideo className="size-4" />
                              </span>
                              <span className="min-w-0 flex-1">
                                <span className="block truncate text-sm font-medium">{j.filename}</span>
                                <span className="block text-xs text-muted-foreground">{formatDate(new Date(j.created_at), language, true)}</span>
                              </span>
                              <Badge className={cn("shrink-0 font-medium", statusTone(j))}>{statusLabel(j)}</Badge>
                            </button>
                          </li>
                        ))}
                      </ol>
                    )}
                  </CardContent>
                </Card>
              </div>
            </>
          ) : (
            <>
              <div className="grid gap-6 lg:grid-cols-[1.5fr_1fr]">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">{t("Новая проверка")}</CardTitle>
                    <CardDescription>{t("Одна запись — полный разбор событий")}</CardDescription>
                  </CardHeader>
                  <CardContent className="flex flex-col gap-4">
                    <div
                      onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
                      onDragLeave={() => setDrag(false)}
                      onDrop={(e) => { e.preventDefault(); setDrag(false); choose(e.dataTransfer.files[0]); }}
                      className={cn(
                        "flex flex-col items-center rounded-lg border-2 border-dashed px-4 py-8 text-center transition-colors",
                        drag ? "border-primary bg-white/[0.08]" : "bg-white/[0.02]",
                      )}
                    >
                      <span className="mb-4 grid size-14 place-items-center rounded-xl bg-accent text-accent-foreground">
                        <UploadCloud className="size-7" />
                      </span>
                      <p className="max-w-full break-words font-medium">{file ? file.name : t("Перетащите видео сюда")}</p>
                      <p className="mb-4 mt-1 text-sm text-muted-foreground">
                        {file ? `${(file.size / 1024 / 1024).toFixed(1)} ${t("МБ · готово к загрузке")}` : t("или выберите файл на вашем устройстве")}
                      </p>
                      <Button variant="outline" size="sm" onClick={() => inputRef.current?.click()} disabled={uploading}>
                        {file ? t("Выбрать другое") : t("Выбрать видео")}
                      </Button>
                      <p className="mt-4 text-xs text-muted-foreground">{t("MP4, MOV, WebM · до 500 МБ")}</p>
                      <input
                        ref={inputRef}
                        type="file"
                        accept=".mp4,.mov,.webm"
                        hidden
                        onChange={(e) => { choose(e.target.files?.[0]); e.target.value = ""; }}
                      />
                    </div>
                    {preview && <video className="max-h-80 w-full rounded-lg bg-black" src={preview} controls />}
                    <Button className="w-full gap-2" disabled={!file || uploading} onClick={upload}>
                      {uploading ? <Loader2 className="size-4 animate-spin" /> : <Play className="size-4" />}
                      {uploading ? t("Загрузка видео…") : t("Начать анализ")}
                    </Button>
                    <p className="flex items-center justify-center gap-1.5 text-xs text-muted-foreground">
                      <LockKeyhole className="size-3.5" />
                      {t("Видео остаётся в локальном контуре")}
                    </p>
                  </CardContent>
                </Card>

                <Card className="hidden lg:block">
                  <CardHeader>
                    <CardTitle className="text-lg">{t("Как проходит проверка")}</CardTitle>
                    <CardDescription>{t("От видео к решению")}</CardDescription>
                  </CardHeader>
                  <CardContent className="flex flex-col gap-5">
                    {[
                      [t("Загрузите запись"), t("Выберите видео работы кассира.")],
                      [t("Дождитесь анализа"), t("Статус обновляется автоматически.")],
                      [t("Проверьте результат"), t("Изучите события, таймкоды и доказательства.")],
                    ].map(([title, text], i) => (
                      <div key={title} className="flex gap-3">
                        <span className="grid size-7 shrink-0 place-items-center rounded-full border text-xs font-medium text-primary">{i + 1}</span>
                        <div>
                          <p className="text-sm font-medium">{title}</p>
                          <p className="mt-0.5 text-sm text-muted-foreground">{text}</p>
                        </div>
                      </div>
                    ))}
                    <div className="flex gap-3 rounded-lg bg-muted p-4 text-sm text-muted-foreground">
                      <ShieldCheck className="size-5 shrink-0 text-primary" />
                      <p>{t("Результат — подсказка для проверки человеком, а не окончательное заключение.")}</p>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {current && (
                <Card>
                  <CardHeader className="flex flex-row items-start justify-between gap-4 space-y-0">
                    <div className="min-w-0 space-y-2">
                      <CardDescription className="text-xs font-medium uppercase tracking-wider">{t("Результат проверки")}</CardDescription>
                      <CardTitle className="break-words text-lg">{current.filename}</CardTitle>
                      <div className="flex flex-wrap items-center gap-2">
                        <Badge className={cn("font-medium", statusTone(current))}>{statusLabel(current)}</Badge>
                        {current.demo && <Badge variant="outline">{t("ДЕМО")}</Badge>}
                      </div>
                    </div>
                    <div className="flex shrink-0 items-center gap-1">
                      {current.status === "done" && (
                        <Button variant="outline" size="sm" className="gap-2" onClick={() => report(current)}>
                          <ArrowDownToLine className="size-4" />
                          <span className="hidden sm:inline">{t("Скачать JSON")}</span>
                        </Button>
                      )}
                      <Button variant="ghost" size="icon" className="size-9" aria-label={t("Закрыть результат")} onClick={() => setSelected(null)}>
                        <X className="size-4" />
                      </Button>
                    </div>
                  </CardHeader>
                  <CardContent className="flex flex-col gap-4">
                    {isActive(current) && (
                      <p className="flex items-center gap-2 text-sm text-muted-foreground">
                        <Loader2 className="size-4 animate-spin text-primary" />
                        {current.status === "queued" ? t("Видео ожидает своей очереди.") : t("Идёт анализ видео. Это может занять несколько минут.")}
                      </p>
                    )}
                    {current.error && <Notice tone="error" icon={<ShieldAlert />}>{t(current.error)}</Notice>}
                    <StoredVideo key={current.job_id + key} id={current.job_id} localURL={localVideos[current.job_id]} videoRef={videoRef} />
                    {current.status === "done" && current.verdict === "no_violation" && (
                      <div className="flex gap-3 rounded-lg bg-success-soft p-4 text-success-foreground">
                        <ShieldCheck className="size-6 shrink-0" />
                        <div>
                          <p className="font-medium">{t("Нарушение не обнаружено")}</p>
                          <p className="mt-1 text-sm opacity-90">{t("В доступной записи не выявлено признаков возможных нарушений.")}</p>
                        </div>
                      </div>
                    )}
                    {current.status === "done" && current.verdict === "violation" && (
                      <div className="flex gap-3 rounded-lg bg-warning-soft p-4 text-warning-foreground">
                        <ShieldAlert className="size-6 shrink-0" />
                        <div>
                          <p className="font-medium">{t("Обнаружены признаки возможных нарушений")}</p>
                          <p className="mt-1 text-sm opacity-90">{t("Проверьте каждое событие по кадрам и таймкоду.")}</p>
                        </div>
                      </div>
                    )}
                    {current.violations.map((v, i) => (
                      <article key={i} className="rounded-lg border p-4">
                        <div className="flex flex-wrap items-center gap-3">
                          <span className="grid size-8 place-items-center rounded-md bg-warning-soft text-sm font-semibold tabular-nums text-warning-foreground">
                            {String(i + 1).padStart(2, "0")}
                          </span>
                          <h3 className="min-w-0 flex-1 font-medium">{v.type}</h3>
                          <Button
                            variant="secondary"
                            size="sm"
                            className="h-8 gap-1.5 tabular-nums"
                            aria-label={`${t("Перейти к")} ${timecode(v.timecode_sec)}`}
                            onClick={() => { if (videoRef.current) videoRef.current.currentTime = v.timecode_sec; }}
                          >
                            <Play className="size-3.5" />
                            {timecode(v.timecode_sec)}
                          </Button>
                        </div>
                        <p className="mt-3 text-sm leading-relaxed text-muted-foreground">{v.description}</p>
                        {/* frames are often portrait (phone camera), so cap the height instead of stretching to the column */}
                        <div className="mt-4 flex flex-wrap gap-3">
                          {v.evidence_frames.map((frame, k) => (
                            <figure key={k} className="m-0 max-w-full">
                              <img
                                className="h-56 w-auto max-w-full rounded-md border object-contain sm:h-72"
                                src={`data:${frame.image_base64.startsWith("iVBOR") ? "image/png" : "image/jpeg"};base64,${frame.image_base64}`}
                                alt={`${current.demo ? t("Синтетическая иллюстрация") : t("Кадр-доказательство")} ${timecode(frame.timestamp_sec)}`}
                              />
                              <figcaption className="mt-1.5 text-xs text-muted-foreground">
                                {current.demo ? t("Демо-иллюстрация") : t("Кадр-доказательство")} · {timecode(frame.timestamp_sec)}
                              </figcaption>
                            </figure>
                          ))}
                        </div>
                        <div className="mt-4 flex gap-2 border-t pt-3 text-xs text-muted-foreground">
                          <ShieldCheck className="size-4 shrink-0 text-primary" />
                          {v.rule ? (
                            <span>
                              {v.rule.title} · {v.rule.section}{" "}
                              {/^https?:\/\//.test(v.rule.url) && (
                                <a className="text-primary underline underline-offset-2" href={v.rule.url} target="_blank" rel="noreferrer">
                                  {t("Открыть источник ↗")}
                                </a>
                              )}
                            </span>
                          ) : (
                            <span>{t("Нормативный источник не предоставлен анализатором.")}</span>
                          )}
                        </div>
                      </article>
                    ))}
                  </CardContent>
                </Card>
              )}

              <Card className="overflow-hidden">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-lg">
                    {t("История анализов")}
                    <Badge variant="secondary" className="tabular-nums">{jobs.length}</Badge>
                  </CardTitle>
                  <CardDescription>{t("Записи и результаты в одном месте")}</CardDescription>
                </CardHeader>
                <div className="flex flex-col gap-3 px-6 pb-4 sm:flex-row sm:items-center sm:justify-between">
                  <div className="flex flex-wrap gap-1" role="group" aria-label={t("Фильтр")}>
                    {filters.map(([id, label]) => (
                      <Button key={id} variant={filter === id ? "secondary" : "ghost"} size="sm" className="h-8" aria-pressed={filter === id} onClick={() => setFilter(id)}>
                        {label}
                      </Button>
                    ))}
                  </div>
                  <div className="relative sm:w-60">
                    <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
                    <Input className="h-9 pl-9" placeholder={t("Найти видео…")} aria-label={t("Поиск видео")} value={search} onChange={(e) => setSearch(e.target.value)} />
                  </div>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full whitespace-nowrap text-left text-sm">
                    <thead className="border-y bg-muted/50 text-xs text-muted-foreground">
                      <tr>
                        <th className="px-6 py-3 font-medium">{t("Видео")}</th>
                        <th className="px-6 py-3 font-medium">{t("Дата загрузки")}</th>
                        <th className="px-6 py-3 font-medium">{t("Статус")}</th>
                        <th className="px-6 py-3 font-medium">{t("События")}</th>
                        <th className="px-6 py-3"><span className="sr-only">{t("Открыть")}</span></th>
                      </tr>
                    </thead>
                    <tbody>
                      {displayed.slice(0, 100).map((j) => (
                        <tr key={j.job_id} onClick={() => open(j)} className={cn("cursor-pointer border-b transition-colors hover:bg-muted/50", j.job_id === selected && "bg-accent/60")}>
                          <td className="px-6 py-3">
                            <span className="flex items-center gap-3">
                              <span className="grid size-9 place-items-center rounded-md bg-muted text-muted-foreground">
                                <FileVideo className="size-4" />
                              </span>
                              <span className="min-w-0">
                                <span className="block max-w-60 truncate font-medium">{j.filename}</span>
                                <span className="block text-xs text-muted-foreground">{j.demo ? t("Демонстрационная проверка") : `ID: ${j.job_id.slice(0, 8)}`}</span>
                              </span>
                            </span>
                          </td>
                          <td className="px-6 py-3 text-muted-foreground">{formatDate(new Date(j.created_at), language, true)}</td>
                          <td className="px-6 py-3"><Badge className={cn("font-medium", statusTone(j))}>{statusLabel(j)}</Badge></td>
                          <td className="px-6 py-3 tabular-nums text-muted-foreground">{j.status === "done" ? j.violations.length : "—"}</td>
                          <td className="px-6 py-3 text-right">
                            <Button variant="ghost" size="icon" className="size-8" aria-label={`${t("Открыть")} ${j.filename}`} onClick={(e) => { e.stopPropagation(); open(j); }}>
                              <ArrowRight className="size-4" />
                            </Button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {displayed.length === 0 && (
                  <div className="p-6">
                    <EmptyState
                      title={search || filter !== "all" ? t("Видео не найдены") : t("Здесь появятся ваши проверки")}
                      text={search || filter !== "all" ? t("Попробуйте изменить поиск или фильтр.") : t("Загрузите первое видео, чтобы начать контроль кассовых операций.")}
                    />
                  </div>
                )}
                <div className="flex items-center justify-between gap-4 px-6 py-3 text-xs text-muted-foreground">
                  <span>{t("Показано")} {Math.min(displayed.length, 100)} {t("из")} {displayed.length}</span>
                  <span className="hidden items-center gap-1.5 sm:flex"><LockKeyhole className="size-3.5" />{t("Данные хранятся локально")}</span>
                </div>
              </Card>
            </>
          )}

          <footer className="flex flex-wrap justify-between gap-2 pt-2 text-xs text-muted-foreground">
            <span>CashGuard · {t("Интеллектуальный контроль операций")}</span>
            <span>{t("Проверка на основе наблюдаемых действий")}</span>
          </footer>
        </main>
      </SidebarInset>
      <AccountDialog open={keyDialog} onClose={() => setKeyDialog(false)} />
    </SidebarProvider>
  );
}

function EmptyState({ title, text, action }: { title: string; text: string; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center py-6 text-center">
      <span className="mb-3 grid size-12 place-items-center rounded-xl bg-muted text-muted-foreground">
        <FileVideo className="size-6" />
      </span>
      <p className="text-sm font-medium">{title}</p>
      <p className="mb-4 mt-1 max-w-xs text-sm text-muted-foreground">{text}</p>
      {action}
    </div>
  );
}

export default App1;
