/**
 * Client-side behaviour for the pool site.
 *
 * There is deliberately very little of it: every number on every page is
 * server-rendered, so this only handles the three things that genuinely depend
 * on the viewer — their theme, their time zone, and motion on first paint.
 *
 * The logic is factored into pure functions with no DOM access so it can be
 * unit tested directly; `init()` is the only part that touches the document.
 */

export const THEME_KEY = 'pool-theme';
export const TZ_KEY = 'pool-tz';
export const ME_KEY = 'pool-me';
export const DETAIL_KEY = 'pool-detail';
export const COMPARE_KEY = 'pool-compare';
export const DEFAULT_TZ = 'America/New_York';
export const DEFAULT_THEME = 'dark';

/**
 * Colours for the comparison chart, assigned in the order people are picked.
 *
 * Pick order rather than a fixed per-entrant colour, so the first person you
 * choose is always the first colour and nobody's line changes hue as the
 * standings move. Six is the cap: beyond that the hues stop being reliably
 * distinguishable, which is the whole reason to have distinct ones.
 */
export const PICK_COLORS = [
  '#c6ff3d', '#4dd8e6', '#ff9f1c', '#b48ce8', '#ff6b35', '#5ee6a8',
];

/**
 * Flip between themes.
 *
 * Dark is the site's default, unconditionally — it does not follow the
 * operating system — so anything that is not an explicit "light" is dark, and
 * the first click always lands on light.
 */
export function nextTheme(current) {
  return current === 'light' ? DEFAULT_THEME : 'light';
}

/**
 * Render a UTC timestamp in the viewer's chosen zone.
 *
 * Falls back to the original string rather than throwing, because a bad stored
 * time zone should never blank out the "data through" stamp.
 */
export function formatTimestamp(iso, timeZone, locale = 'en-US') {
  const when = new Date(iso);
  if (Number.isNaN(when.getTime())) return iso;
  try {
    return new Intl.DateTimeFormat(locale, {
      timeZone,
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
      timeZoneName: 'short',
    }).format(when);
  } catch {
    return iso;
  }
}

/** Is this a time zone the browser will accept? */
export function isValidTimeZone(tz) {
  if (!tz) return false;
  try {
    new Intl.DateTimeFormat('en-US', { timeZone: tz });
    return true;
  } catch {
    return false;
  }
}

export function easeOutCubic(t) {
  const clamped = Math.min(Math.max(t, 0), 1);
  return 1 - Math.pow(1 - clamped, 3);
}

/** Value part-way through a count-up, used by the scoreboard odometer. */
export function countUpValue(target, progress) {
  return target * easeOutCubic(progress);
}

/**
 * Push overlapping endpoint labels apart, keeping their vertical order.
 *
 * Two people finishing a week on the same score put their labels in exactly the
 * same place. The markers stay on the true values; only the text moves.
 */
export function spreadLabels(items, gap = 14, top = -Infinity, bottom = Infinity) {
  let previous = -Infinity;
  const placed = [...items]
    .sort((a, b) => a.y - b.y)
    .map(({ slug, y }) => {
      const yOut = Math.max(y, previous + gap);
      previous = yOut;
      return { slug, y: yOut };
    });

  // If the stack overflowed the plot, slide the whole run back up — the same
  // correction the Python `_spread` has always applied. Without it, labels for
  // lines finishing near the chart bottom were pushed past the viewBox and
  // clipped invisible.
  if (placed.length && bottom < Infinity && placed[placed.length - 1].y > bottom) {
    const shift = Math.min(placed[placed.length - 1].y - bottom, placed[0].y - top);
    if (shift > 0) return placed.map((p) => ({ slug: p.slug, y: p.y - shift }));
  }
  return placed;
}

/**
 * Order two table cells.
 *
 * Numeric when both sides genuinely are numbers, lexical otherwise — so a
 * points column sorts 9 before 10, while a "0-0" record column falls back to
 * text instead of being silently read as zero.
 */
export function compareValues(a, b) {
  const left = String(a).trim();
  const right = String(b).trim();
  const nl = Number(left);
  const nr = Number(right);
  if (left !== '' && right !== '' && Number.isFinite(nl) && Number.isFinite(nr)) {
    return nl - nr;
  }
  return left.localeCompare(right, 'en', { numeric: true, sensitivity: 'base' });
}

/** The value a cell sorts on: an explicit data-value if present, else its text. */
export function cellValue(row, index) {
  const cell = row.children[index];
  if (!cell) return '';
  const explicit = cell.getAttribute('data-value');
  return explicit === null ? cell.textContent : explicit;
}

/** Reorder a table body in place. Returns the rows in their new order. */
export function sortTable(table, index, direction) {
  const body = table.tBodies[0];
  if (!body) return [];
  const rows = Array.from(body.rows);
  rows.sort((a, b) => direction * compareValues(cellValue(a, index), cellValue(b, index)));
  for (const row of rows) body.appendChild(row);
  return rows;
}

/** Storage that silently no-ops when it is unavailable (private browsing). */
export function safeStorage(store) {
  return {
    get(key) {
      try {
        return store.getItem(key);
      } catch {
        return null;
      }
    },
    set(key, value) {
      try {
        store.setItem(key, value);
      } catch {
        /* nothing sensible to do */
      }
    },
  };
}

// ---------------------------------------------------------------------------
// DOM wiring
// ---------------------------------------------------------------------------
function initTheme(doc, storage) {
  const button = doc.querySelector('[data-theme-toggle]');
  if (!button) return;
  button.addEventListener('click', () => {
    const theme = nextTheme(doc.documentElement.dataset.theme);
    doc.documentElement.dataset.theme = theme;
    storage.set(THEME_KEY, theme);
  });
}

/**
 * The Details toggle on the standings page.
 *
 * The board's default is the glance: rank, name, teams, points. The outlook
 * bar, the sparkline and the ceiling maths appear for whoever asks, and the
 * choice is remembered per browser. `aria-pressed` tracks the state so the
 * control reads correctly to a screen reader.
 */
function initDetail(doc, storage) {
  const button = doc.querySelector('[data-detail-toggle]');
  if (!button) return;

  const paint = () => {
    button.setAttribute(
      'aria-pressed',
      String(doc.documentElement.dataset.detail === 'on'),
    );
  };
  paint();

  button.addEventListener('click', () => {
    const on = doc.documentElement.dataset.detail === 'on';
    if (on) delete doc.documentElement.dataset.detail;
    else doc.documentElement.dataset.detail = 'on';
    storage.set(DETAIL_KEY, on ? 'off' : 'on');
    paint();
  });
}

/**
 * "This is me": one stored slug that highlights the viewer everywhere.
 *
 * Six near-identical rows on a phone is exactly where finding yourself is the
 * whole job. The choice never leaves the browser — there is no server here to
 * send it to.
 */
/**
 * The viewer's own line, lifted to the top of the standings.
 *
 * Built from the board row that already exists — the strip never adds entrant
 * links of its own, it borrows the row's. Empty (and hidden) until "who are
 * you?" is set, and on every page that has no board.
 */
function paintYouStrip(doc, slug) {
  const strip = doc.querySelector('[data-you]');
  if (!strip) return;

  const row = slug ? doc.querySelector(`.board .row[data-slug="${slug}"]`) : null;
  strip.textContent = '';
  strip.hidden = !row;
  if (!row) return;

  const grab = (sel) => {
    const el = row.querySelector(sel);
    return el ? el.textContent.trim() : '';
  };

  const link = doc.createElement('a');
  link.href = row.getAttribute('href');

  const rank = doc.createElement('span');
  rank.className = 'you-rank';
  rank.textContent = `#${grab('.row-rank')}`;

  const label = doc.createElement('span');
  label.className = 'you-label';
  label.textContent = 'your entry';

  const name = doc.createElement('span');
  name.textContent = grab('.row-name');

  const pts = doc.createElement('span');
  pts.className = 'you-pts';
  pts.textContent = `${grab('.row-points')} pts`;

  link.append(rank, label, name, pts);
  strip.append(link);
}

function initMe(doc, storage) {
  const select = doc.querySelector('[data-me-select]');
  const stored = storage.get(ME_KEY) || '';

  const apply = (slug) => {
    doc.documentElement.dataset.me = slug;
    for (const el of doc.querySelectorAll('[data-slug]')) {
      el.classList.toggle('is-me', Boolean(slug) && el.dataset.slug === slug);
    }
    paintYouStrip(doc, slug);
  };

  if (select) {
    // A stored name that is no longer in the pool must not leave the picker
    // showing a blank: next season's picks file will not have last year's
    // entrants in it.
    select.value = stored;
    const slug = select.value === stored ? stored : '';
    apply(slug);
    if (slug !== stored) storage.set(ME_KEY, slug);

    select.addEventListener('change', () => {
      storage.set(ME_KEY, select.value);
      apply(select.value);
    });
    return;
  }
  apply(stored);
}

/**
 * The comparison chart: pick who you want to read against.
 *
 * Every line is already in the document; picking only toggles classes and sets
 * a colour, so with scripting off the chart still renders the whole field.
 */
function initCompare(doc, storage) {
  const charts = Array.from(doc.querySelectorAll('[data-compare]'));
  const buttons = Array.from(doc.querySelectorAll('[data-pick]'));
  if (!charts.length || !buttons.length) return;

  const known = new Set(buttons.map((b) => b.dataset.pick));
  const stored = (storage.get(COMPARE_KEY) || '').split(' ').filter(Boolean);
  const me = doc.documentElement.dataset.me;

  // Nothing stored yet, but they have told us who they are: start on them.
  // Opening the page already showing your own line is the point of the chart.
  let picked = (stored.length ? stored : (me ? [me] : [])).filter((s) => known.has(s));

  const empty = doc.querySelector('[data-compare-empty]');

  const paint = () => {
    const colors = new Map(
      picked.map((slug, i) => [slug, PICK_COLORS[i % PICK_COLORS.length]]),
    );

    for (const button of buttons) {
      const color = colors.get(button.dataset.pick);
      button.setAttribute('aria-pressed', String(Boolean(color)));
      if (color) button.style.setProperty('--pick-color', color);
      else button.style.removeProperty('--pick-color');
    }

    for (const chart of charts) {
      for (const el of chart.querySelectorAll('[data-entrant]')) {
        const color = colors.get(el.dataset.entrant);
        el.classList.toggle('is-picked', Boolean(color));
        if (color) el.style.setProperty('--pick-color', color);
        else el.style.removeProperty('--pick-color');
      }

      const labels = Array.from(chart.querySelectorAll('.cmp-label.is-picked'));
      // Bounds from the chart's own viewBox (14 top pad, 26 bottom pad —
      // matching the Python renderer), so an overflowing label stack slides
      // back into view instead of clipping.
      const box = chart.querySelector('svg')?.viewBox?.baseVal;
      const placed = spreadLabels(
        labels.map((el) => ({ slug: el.dataset.entrant, y: Number(el.dataset.y) })),
        14,
        box ? 14 : -Infinity,
        box ? box.height - 26 : Infinity,
      );
      for (const { slug, y } of placed) {
        const el = chart.querySelector(`.cmp-label[data-entrant="${slug}"]`);
        if (el) el.setAttribute('y', String(y + 4));
      }
    }

    if (empty) empty.hidden = picked.length > 0;
    storage.set(COMPARE_KEY, picked.join(' '));
  };

  for (const button of buttons) {
    button.addEventListener('click', () => {
      const slug = button.dataset.pick;
      picked = picked.includes(slug)
        ? picked.filter((s) => s !== slug)
        // Past the palette, the oldest pick drops out rather than two lines
        // sharing a colour — which would defeat the entire chart.
        : [...picked, slug].slice(-PICK_COLORS.length);
      paint();
    });
  }
  paint();
}

/** Click (or Enter/Space) a marked column heading to sort the table by it. */
function initSort(doc) {
  for (const table of doc.querySelectorAll('table')) {
    const heads = Array.from(table.querySelectorAll('th[data-sort]'));
    if (!heads.length) continue;

    for (const th of heads) {
      th.tabIndex = 0;
      const activate = () => {
        const index = Array.from(th.parentElement.children).indexOf(th);
        // First click on a column sorts the way that column is most useful:
        // biggest-first for numbers (headings the templates mark `num`),
        // A-Z for names. Previously this comment described behaviour the
        // code never had — every column opened ascending.
        const current = th.getAttribute('aria-sort');
        let direction;
        if (current === null) {
          direction = th.classList.contains('num') ? -1 : 1;
        } else {
          direction = current === 'ascending' ? -1 : 1;
        }

        for (const other of heads) other.removeAttribute('aria-sort');
        th.setAttribute('aria-sort', direction === 1 ? 'ascending' : 'descending');
        sortTable(table, index, direction);
      };

      th.addEventListener('click', activate);
      th.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          activate();
        }
      });
    }
  }
}

function initTimeZone(doc, storage) {
  const select = doc.querySelector('[data-tz-select]');
  const stamps = Array.from(doc.querySelectorAll('time[data-ts]'));
  if (!stamps.length) return;

  const stored = storage.get(TZ_KEY);
  let zone = isValidTimeZone(stored) ? stored : DEFAULT_TZ;

  const paint = () => {
    for (const el of stamps) {
      const iso = el.getAttribute('datetime');
      if (iso) el.textContent = formatTimestamp(iso, zone);
    }
  };

  if (select) {
    // A valid stored zone that is not one of the listed options (an old
    // build's list, or a hand-set value) still applies to every timestamp;
    // only the <select> shows blank. The previous code did the opposite of
    // its own comment — it read the emptied select back and snapped the
    // viewer to Eastern.
    select.value = zone;
    select.addEventListener('change', () => {
      zone = select.value;
      storage.set(TZ_KEY, zone);
      paint();
    });
  }
  paint();
}

function initOdometer(doc, win) {
  if (win.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  const targets = Array.from(doc.querySelectorAll('.row-points, .hero-total'));
  const runs = targets
    .map((el) => ({ el, value: Number.parseFloat(el.textContent) }))
    .filter((t) => Number.isFinite(t.value) && t.value > 0);
  if (!runs.length) return;

  const duration = 620;
  const start = win.performance.now();
  const settle = () => {
    for (const { el, value } of runs) el.textContent = value.toFixed(2);
  };

  // If animation frames stop arriving — a backgrounded tab, a throttled
  // device — the count-up must not strand a half-counted number on screen.
  // This guarantees the true value regardless of what the frames do.
  win.setTimeout(settle, duration + 400);

  const step = (now) => {
    const progress = (now - start) / duration;
    if (progress >= 1) {
      settle();
      return;
    }
    for (const { el, value } of runs) {
      el.textContent = countUpValue(value, progress).toFixed(2);
    }
    win.requestAnimationFrame(step);
  };
  win.requestAnimationFrame(step);
}

export function init(doc = document, win = window) {
  const storage = safeStorage(win.localStorage);
  initTheme(doc, storage);
  initDetail(doc, storage);
  // Identity first: the comparison chart opens on whoever the viewer says
  // they are, so it has to know before it paints.
  initMe(doc, storage);
  initTimeZone(doc, storage);
  initCompare(doc, storage);
  initSort(doc);
  initOdometer(doc, win);
}

if (typeof document !== 'undefined' && typeof window !== 'undefined') {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => init());
  } else {
    init();
  }
}
