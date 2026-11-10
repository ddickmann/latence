import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

// jsdom has no layout, so nothing ever resizes; Carbon's text area asks for the observer anyway.
class NoResize {
  observe(): void {}
  unobserve(): void {}
  disconnect(): void {}
}
globalThis.ResizeObserver ??= NoResize as unknown as typeof ResizeObserver;

afterEach(() => cleanup());
