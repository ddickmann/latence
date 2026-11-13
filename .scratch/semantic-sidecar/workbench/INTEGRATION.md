# Mounting the workbench in a Carbon host

The workbench is built to be compiled into a host application's own bundle and mounted as one
route in the host's router. It brings no second React, no second router and no iframe. The
host imports the workbench; the workbench imports nothing from the host.

This page lists what a mount needs. Nothing here has been done in any host yet. The claims are
backed by `src/test/mount.test.tsx`, which mounts the component inside a router it does not
own, at a path it did not choose, with a host's chrome and a host's own button.

## What the host must already have

| Package | Version this was built and tested against |
|---|---|
| `react`, `react-dom` | 19.2.0 |
| `react-router` | 7.18.4 |
| `@carbon/react` | 1.92.0 |
| `@carbon/icons-react` | 11.90.0 (any `^11.64.0`) |

They are peer dependencies. The library build (`dist/resolution-workbench.js`, 101 kB) leaves
all of them out, so the host's copies are the only ones on the page.

## The mount, in five edits

### 1. A route

```tsx
import { ResolutionWorkbench } from "resolution-workbench";
import "resolution-workbench/styles.css";

<Route path="/resolution/*" element={<ResolutionPage />} />
```

The trailing `/*` is required: the workbench reads the rest of the path itself
(`review`, `review/<pair>`, `graph`, `graph/<node>`) and builds its links from the path it
finds itself mounted at. The open pair or node is therefore part of the host's URL, and the
host's back button, links and bookmarks work.

### 2. A page component that passes the host's seams

```tsx
function ResolutionPage() {
  const { recount } = useChrome();          // the host's own hooks
  const toast = useToast();
  const { t } = useTranslation();
  const chrome = useMemo(() => ({ useCrumbs, useListKeys, recount, notify: toast }), [recount, toast]);
  return (
    <ResolutionWorkbench
      reviewSource={reviewSource}
      graphSource={graphSource}
      chrome={chrome}
      controls={{ Button, TextField, TextArea: TextAreaField, Select: SelectField, Checkbox: CheckField }}
      translate={(key, vars) => t(`resolution.${key}`, vars)}
      locale={i18n.language}
    />
  );
}
```

Every prop except `reviewSource` is optional and has a default, so a mount can start with the
route alone and take the seams one at a time.

| Seam | What the workbench calls | Default without it |
|---|---|---|
| `chrome.useCrumbs(crumbs)` | on every render, with `{ label, to? }[]` | no breadcrumb |
| `chrome.useListKeys({ next, previous, close })` | on every render; the host owns J, K and Esc and calls these | the workbench listens for J, K and Esc on the window |
| `chrome.recount()` | after each recorded decision | nothing |
| `chrome.notify({ tone, title, detail?, action? })` | after each recorded decision or failure; a group decision passes an `action` (`{ label, run }`) for undo | the workbench shows its own notice |
| `controls.Button` etc. | wherever it needs one of the five controls | Carbon's components, unstyled beyond Carbon |
| `translate(key, vars)` | for every string | the bundled English |

`useCrumbs` and `useListKeys` are called as hooks, so pass the host's hooks themselves and
keep them the same functions for the life of the mount. The decision keys (`A`, `R`, `U`,
`Space`, `Y` for the suggestion, and `M` and `S` for the whole group) are always the
workbench's own; they are ignored while a person types in a field and
while a modifier key is held.

The control props are deliberately small (`src/host.ts`): `Button` takes `kind`
(`primary | secondary | quiet | danger`), `size`, `kbd`, `disabled`, `onClick`, `testId`;
the fields take `id`, `label`, `value`, `onChange(value)`. A host whose own wrappers accept at
least these can be passed as they are.

### 3. Messages

`resolution-workbench/messages/en.json` holds every string in i18next's `{{name}}` form, with
`_one` / `_other` plurals. Merge it into the host's catalogue under a prefix (`resolution` in
the example above). German follows with the German phase; until then a host can fall back to
English for this prefix.

### 4. A navigation entry

One item in the host's navigation that points at `/resolution`. To show the number of open
pairs there, read it from the same review source; `chrome.recount()` tells the host when it
changed.

### 5. Theme: one small stylesheet

The workbench styles itself only through `--rw-*` custom properties on `.rw-root`
(`src/styles/workbench.scss`, top of file). Their defaults follow Carbon's theme variables
where the page defines them (`--cds-background`, `--cds-text-primary`, …) and Carbon's White
and Gray 100 values where it does not, switching on `html[data-theme="dark"]` and on the
system setting. A host with its own tokens maps them once:

```css
.rw-root {
  --rw-bg: var(--your-surface);
  --rw-layer: var(--your-surface-raised);
  --rw-text: var(--your-ink);
  --rw-text-muted: var(--your-ink-muted);
  --rw-border: var(--your-line);
  --rw-accent: var(--your-accent);
  --rw-focus: var(--your-focus);
  --rw-font: var(--your-font-sans);
  --rw-font-mono: var(--your-font-mono);
  --rw-radius: var(--your-radius);
  --rw-offset: 48px;   /* what sits above the workbench: the host's header */
}
```

The full list is short (22 colours, two fonts, a radius, an offset). The map lives in the
host; no host token value or class is copied into this package. All class names start with
`rw-`, so they cannot collide with the host's.

On a wide screen the workbench fills the window height below `--rw-offset` and scrolls its
panes separately. Set `--rw-height` instead if the host gives it a box of its own.

## Data

The workbench talks to two interfaces (`src/dataSource.ts`):

- `ReviewDataSource`: `loadQueue`, `loadLedger`, `decide`, `decideGroup`, `withdrawBatch`,
  `loadAudit`.
- `GraphDataSource`: `loadSchema`, `loadNode`, `search`, `listInstances`, `entityOfRecord`.

The prototype ships `StaticReviewSource` and `StaticGraphSource`, which serve an export file
from memory and keep the decision log in the browser. A deployment replaces them with a client
for its own service: the review state in Delta (ADR-0072) behind a small API, or a host's
backend-for-frontend. The interfaces are asynchronous throughout for that reason.
`decideGroup` is one call on purpose: a service appends the group record and all its pair
decisions in one transaction, under one batch id, and `withdrawBatch` appends the withdrawals
for what still stands. A review item may carry a `suggestion` and the queue's `meta` a
`suggester` summary (`src/types.ts`); a source without suggestions leaves both out and the
card does not appear. The rules for which pairs a group decision touches are in
`src/groupActions.ts` (`planMerge`, `planSplit`), pure functions a service can port or call. Two things a
real source has to add that the static one does not need: paging of the queue, and the signed-in
person as the actor on each decision (the static source is given a fixed one).

## Security posture

- No iframe, no remote module, no `eval`, no inline script: the workbench is ordinary modules
  compiled by the host, so a `script-src 'self'` policy holds.
- No network call of its own. Fonts and data come from the host.
- The standalone build and the single-file preview are for looking at the prototype; they are
  not what a host ships.

## Getting the package into a host

Until it is published, build it and install the tarball:

```bash
npm run build:lib && npm pack        # resolution-workbench-0.0.1.tgz
```

A host that pins every dependency adds that tarball like any other package, with its own
review.

## What a mount still needs deciding

- Where the review state lives for the host (its own service or the engine's).
- Whether the graph view is a tab inside the workbench, as here, or its own navigation entry.
  Both URLs exist (`…/review`, `…/graph`), so either works without a change here.
- Roles: who may decide, who may only read. The workbench has no notion of roles yet.
