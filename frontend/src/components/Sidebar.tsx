import {
  Hand,
  Home,
  Languages,
  History,
  Mic2,
  Settings,
  HelpCircle,
  Info,
  HeartHandshake,
  ChevronRight,
} from "lucide-react";

export type Page =
  | "home"
  | "live"
  | "history"
  | "voices"
  | "settings"
  | "help"
  | "about";

const NAV: { id: Page; label: string; icon: React.ReactNode }[] = [
  { id: "home", label: "Home", icon: <Home size={20} strokeWidth={2} /> },
  { id: "live", label: "Live Translate", icon: <Languages size={20} strokeWidth={2} /> },
  { id: "history", label: "History", icon: <History size={20} strokeWidth={2} /> },
  { id: "voices", label: "Voices", icon: <Mic2 size={20} strokeWidth={2} /> },
  { id: "settings", label: "Settings", icon: <Settings size={20} strokeWidth={2} /> },
  { id: "help", label: "Help", icon: <HelpCircle size={20} strokeWidth={2} /> },
  { id: "about", label: "About", icon: <Info size={20} strokeWidth={2} /> },
];

type Props = {
  page: Page;
  onNavigate: (p: Page) => void;
};

export function Sidebar({ page, onNavigate }: Props) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-icon">
          <Hand size={24} strokeWidth={2.2} />
        </div>
        <span className="brand-text">
          Sign Language
          <br />
          to Speech
        </span>
      </div>

      <nav className="nav">
        {NAV.map((item) => {
          const isActive =
            item.id === page ||
            (item.id === "home" && (page === "home" || page === "live"));
          return (
            <button
              key={item.id}
              type="button"
              className={`nav-item ${isActive ? "active" : ""}`}
              onClick={() => onNavigate(item.id === "live" ? "live" : item.id)}
            >
              {item.icon}
              {item.label}
            </button>
          );
        })}
      </nav>

      <div className="access-card">
        <HeartHandshake className="access-card-icon" size={20} strokeWidth={2} />
        <strong>Enable Accessibility</strong>
        <p>Larger text &amp; high contrast for easier use in public spaces.</p>
        <ChevronRight className="access-card-arrow" size={16} strokeWidth={2.5} />
      </div>
    </aside>
  );
}
