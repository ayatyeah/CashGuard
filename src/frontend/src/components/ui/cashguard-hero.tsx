import { Plus, ShieldCheck } from "lucide-react";
import { useWorkspace } from "@/lib/workspace";

export function CashGuardHero({ onAnalyze }: { onAnalyze: () => void }) {
  const { t } = useWorkspace();
  return (
    <section className="cashguard-hero" aria-labelledby="cashguard-hero-title">
      <div className="cashguard-hero-content">
        <p className="cashguard-eyebrow">FORTE · {t("ВИДЕОАНАЛИТИКА")}</p>
        <h2 id="cashguard-hero-title">CashGuard</h2>
        <p className="cashguard-subtitle">{t("Кассовые операции под контролем")}</p>
        <p className="cashguard-description">{t("Анализ видео, таймкоды и кадры для проверки.")}</p>
        <div className="cashguard-actions">
          <button type="button" className="cashguard-cta" onClick={onAnalyze}>
            <Plus size={18} aria-hidden />{t("Новый анализ")}
          </button>
          <span className="cashguard-local"><ShieldCheck size={15} aria-hidden />{t("Локальный контур")}</span>
        </div>
      </div>
      <img className="cashguard-hero-mark" src="/brand/forte-hero.svg" alt="" aria-hidden="true" />
    </section>
  );
}
