import { isTauri } from "@tauri-apps/api/core";
import { load, type Store } from "@tauri-apps/plugin-store";
import type { StateStorage } from "zustand/middleware";
import { debounce } from "@/hooks/use-debounce";

const hasBrowserWindow = () => typeof window !== "undefined";

export class TauriStoreState implements StateStorage {
  private store: Store | null = null;
  private debouncedSave: (() => void) | null = null;
  private initialized = false;

  constructor(public storeName: string) {}

  async init() {
    if (this.initialized) {
      return;
    }

    this.initialized = true;

    if (!hasBrowserWindow()) {
      // When server-rendering we skip initializing the Tauri store.
      return;
    }

    if (!isTauri()) {
      // Running in a regular browser; fall back to default persist storage.
      return;
    }

    this.store = await load(this.storeName);
    if (!this.store) {
      throw new Error(`Failed to load store: ${this.storeName}`);
    }

    this.debouncedSave = debounce(() => this.store?.save(), 1 * 1000) ?? null;
  }

  getItem(name: string): string | null {
    if (this.store) {
      // Tauri async path — Zustand persist calls this synchronously first,
      // then calls it again after rehydration; for Tauri we return null on
      // the first sync call and let persist rehydrate via onRehydrateStorage.
      return localStorage.getItem(name);
    }
    return localStorage.getItem(name);
  }

  setItem(name: string, value: string): void {
    localStorage.setItem(name, value);
    if (this.store) {
      void this.store.set(name, value);
      this.debouncedSave?.();
    }
  }

  removeItem(name: string): void {
    localStorage.removeItem(name);
    if (this.store) {
      void this.store.delete(name);
      this.debouncedSave?.();
    }
  }
}
