import { useCallback, useEffect, useState } from "react";
import { Bookmark, Trash2 } from "lucide-react";
import {
  clearHistory,
  deleteHistoryEntry,
  formatHistoryTime,
  loadHistory,
  type HistoryEntry,
} from "../lib/history";

type Props = {
  /** Bump to reload when navigating to History after a save */
  refreshKey?: number;
};

export function HistoryPage({ refreshKey = 0 }: Props) {
  const [entries, setEntries] = useState<HistoryEntry[]>([]);

  const reload = useCallback(() => {
    setEntries(loadHistory());
  }, []);

  useEffect(() => {
    reload();
  }, [reload, refreshKey]);

  const onDelete = (id: string) => {
    setEntries(deleteHistoryEntry(id));
  };

  const onClearAll = () => {
    if (!entries.length) return;
    if (!confirm("Clear all saved conversations?")) return;
    clearHistory();
    setEntries([]);
  };

  return (
    <div className="history-page">
      <header className="page-header">
        <div className="page-header-text">
          <h1>History</h1>
          <p>Conversations you saved from Live Translate</p>
        </div>
        {entries.length > 0 && (
          <button type="button" className="btn-text-sm" onClick={onClearAll}>
            Clear all
          </button>
        )}
      </header>

      {entries.length === 0 ? (
        <div className="history-empty card">
          <Bookmark size={28} strokeWidth={1.75} />
          <strong>No saved conversations yet</strong>
          <p>
            On Live Translate, tap <em>Save in history</em> next to Show Original
            (Raw) to keep a sentence here.
          </p>
        </div>
      ) : (
        <ul className="history-list">
          {entries.map((e) => (
            <li key={e.id} className="card history-item">
              <div className="history-item-main">
                <p className="history-corrected">{e.corrected}</p>
                <p className="history-raw">Raw: {e.raw}</p>
                <time className="history-time" dateTime={new Date(e.savedAt).toISOString()}>
                  {formatHistoryTime(e.savedAt)}
                </time>
              </div>
              <button
                type="button"
                className="history-delete"
                onClick={() => onDelete(e.id)}
                aria-label="Delete conversation"
                title="Delete"
              >
                <Trash2 size={16} />
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
