import { useWorkspace } from '@/lib/workspace';
import type { Language } from '@/lib/translations';

export function Preferences() {
  const { t, language, preferences, saving, ready } = useWorkspace();
  return (
    <select
      aria-label={t('Язык интерфейса')}
      value={language}
      disabled={saving || !ready}
      onChange={(e) => void preferences(e.target.value as Language)}
      className="h-9 rounded-md border border-input bg-white/[0.03] px-2 text-sm text-foreground backdrop-blur focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50"
    >
      <option value="ru">Русский</option>
      <option value="en">English</option>
      <option value="kk">Қазақша</option>
    </select>
  );
}
