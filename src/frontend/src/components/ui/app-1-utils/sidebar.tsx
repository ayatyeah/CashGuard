import * as React from "react";
import { PanelLeft } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTitle } from "@/components/ui/sheet";
import { cn } from "@/lib/utils";

// Minimal shadcn-style sidebar: a permanent column from md up, a left sheet on phones.
const MOBILE_QUERY = "(max-width: 767px)";

type SidebarContextValue = {
  isMobile: boolean;
  open: boolean;
  openMobile: boolean;
  setOpenMobile: (open: boolean) => void;
  toggle: () => void;
};

const SidebarContext = React.createContext<SidebarContextValue | null>(null);

export function useSidebar() {
  const context = React.useContext(SidebarContext);
  if (!context) throw new Error("useSidebar must be used within <SidebarProvider />");
  return context;
}

export function SidebarProvider({ children }: { children: React.ReactNode }) {
  const [isMobile, setIsMobile] = React.useState(() => window.matchMedia(MOBILE_QUERY).matches);
  const [open, setOpen] = React.useState(true);
  const [openMobile, setOpenMobile] = React.useState(false);

  React.useEffect(() => {
    const query = window.matchMedia(MOBILE_QUERY);
    const update = () => setIsMobile(query.matches);
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, []);

  const toggle = React.useCallback(
    () => (isMobile ? setOpenMobile((v) => !v) : setOpen((v) => !v)),
    [isMobile],
  );

  return (
    <SidebarContext.Provider value={{ isMobile, open, openMobile, setOpenMobile, toggle }}>
      <div className="flex min-h-svh w-full bg-background text-foreground">{children}</div>
    </SidebarContext.Provider>
  );
}

export function Sidebar({ title, children }: { title: string; children: React.ReactNode }) {
  const { isMobile, open, openMobile, setOpenMobile } = useSidebar();

  if (isMobile) {
    return (
      <Sheet open={openMobile} onOpenChange={setOpenMobile}>
        <SheetContent side="left" aria-describedby={undefined} className="flex w-72 flex-col border-r bg-sidebar/80 p-0 backdrop-blur-xl">
          <SheetTitle className="sr-only">{title}</SheetTitle>
          {children}
        </SheetContent>
      </Sheet>
    );
  }

  return (
    <>
      {/* spacer keeps the inset next to the fixed column and animates with it */}
      <div aria-hidden className={cn("shrink-0 transition-[width] duration-200", open ? "w-64" : "w-0")} />
      <aside
        aria-label={title}
        className={cn(
          "fixed inset-y-0 left-0 z-20 flex w-64 flex-col border-r bg-sidebar/60 backdrop-blur-xl transition-transform duration-200",
          !open && "-translate-x-full",
        )}
      >
        {children}
      </aside>
    </>
  );
}

export function SidebarInset({ className, ...props }: React.ComponentProps<"div">) {
  return <div className={cn("flex min-h-svh flex-1 flex-col", className)} {...props} />;
}

export function SidebarTrigger(props: React.ComponentProps<typeof Button>) {
  const { toggle } = useSidebar();
  return (
    <Button variant="ghost" size="icon" className="size-9 shrink-0" onClick={toggle} {...props}>
      <PanelLeft className="size-5" />
    </Button>
  );
}

export function SidebarMenuButton({
  active,
  className,
  ...props
}: React.ComponentProps<"button"> & { active?: boolean }) {
  return (
    <button
      aria-current={active ? "page" : undefined}
      className={cn(
        "flex w-full items-center gap-3 rounded-md px-3 py-2 text-left text-sm text-muted-foreground transition-colors hover:bg-sidebar-accent hover:text-sidebar-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring [&>svg]:size-4 [&>svg]:shrink-0",
        active && "bg-sidebar-accent font-medium text-sidebar-accent-foreground",
        className,
      )}
      {...props}
    />
  );
}
