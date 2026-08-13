/**
 * Tests for the site's client-side behaviour.
 *
 * The pure helpers are tested directly; `init()` is exercised against a jsdom
 * document that mirrors the real markup, so the wiring is covered too.
 */

import { beforeEach, describe, expect, test, vi } from 'vitest';

import {
  COMPARE_KEY,
  DEFAULT_THEME,
  DEFAULT_TZ,
  ME_KEY,
  ME_LIMIT,
  PICK_COLORS,
  THEME_KEY,
  TZ_KEY,
  cellValue,
  compareValues,
  countUpValue,
  easeOutCubic,
  formatTimestamp,
  init,
  isValidTimeZone,
  nextTheme,
  safeStorage,
  sortTable,
  spreadLabels,
} from '../../assets/site.js';

const ISO = '2026-02-09T14:30:00+00:00';

describe('nextTheme', () => {
  test('flips an explicit theme', () => {
    expect(nextTheme('dark')).toBe('light');
    expect(nextTheme('light')).toBe('dark');
  });

  test('dark is the default, so anything unset flips to light', () => {
    // The site does not follow the operating system: dark is the default
    // unconditionally, so the first click always lands on light no matter
    // what the viewer's OS is set to.
    expect(DEFAULT_THEME).toBe('dark');
    expect(nextTheme(undefined)).toBe('light');
    expect(nextTheme('')).toBe('light');
    expect(nextTheme('nonsense')).toBe('light');
  });
});

describe('formatTimestamp', () => {
  test('renders a UTC instant in the requested zone', () => {
    const eastern = formatTimestamp(ISO, 'America/New_York');
    expect(eastern).toContain('9:30');
    expect(eastern).toContain('EST');
  });

  test('the same instant differs by zone', () => {
    const eastern = formatTimestamp(ISO, 'America/New_York');
    const pacific = formatTimestamp(ISO, 'America/Los_Angeles');
    expect(eastern).not.toBe(pacific);
    expect(pacific).toContain('6:30');
  });

  test('falls back to the raw string rather than blanking the stamp', () => {
    expect(formatTimestamp('not-a-date', 'UTC')).toBe('not-a-date');
    expect(formatTimestamp(ISO, 'Mars/Olympus')).toBe(ISO);
  });
});

describe('isValidTimeZone', () => {
  test.each([
    ['America/New_York', true],
    ['UTC', true],
    ['Mars/Olympus', false],
    ['', false],
    [null, false],
    [undefined, false],
  ])('%s -> %s', (tz, expected) => {
    expect(isValidTimeZone(tz)).toBe(expected);
  });
});

describe('count-up maths', () => {
  test('easing is clamped and monotonic', () => {
    expect(easeOutCubic(-1)).toBe(0);
    expect(easeOutCubic(0)).toBe(0);
    expect(easeOutCubic(1)).toBe(1);
    expect(easeOutCubic(2)).toBe(1);
    expect(easeOutCubic(0.25)).toBeLessThan(easeOutCubic(0.75));
  });

  test('the count-up starts at zero and lands exactly on the target', () => {
    expect(countUpValue(42.5, 0)).toBe(0);
    expect(countUpValue(42.5, 1)).toBeCloseTo(42.5, 10);
    expect(countUpValue(42.5, 0.5)).toBeGreaterThan(0);
    expect(countUpValue(42.5, 0.5)).toBeLessThan(42.5);
  });
});

describe('spreadLabels', () => {
  test('leaves labels that already clear each other alone', () => {
    const out = spreadLabels([{ slug: 'a', y: 10 }, { slug: 'b', y: 90 }], 14);
    expect(out).toEqual([{ slug: 'a', y: 10 }, { slug: 'b', y: 90 }]);
  });

  test('pushes a collision apart by exactly the gap', () => {
    const out = spreadLabels([{ slug: 'a', y: 50 }, { slug: 'b', y: 50 }], 14);
    expect(out).toEqual([{ slug: 'a', y: 50 }, { slug: 'b', y: 64 }]);
  });

  test('a whole stack on one value fans out in order', () => {
    const out = spreadLabels(
      [{ slug: 'a', y: 20 }, { slug: 'b', y: 22 }, { slug: 'c', y: 24 }],
      14,
    );
    expect(out.map((o) => o.y)).toEqual([20, 34, 48]);
  });

  test('output is sorted top to bottom regardless of input order', () => {
    const out = spreadLabels([{ slug: 'z', y: 80 }, { slug: 'a', y: 10 }], 14);
    expect(out.map((o) => o.slug)).toEqual(['a', 'z']);
  });

  test('does not mutate the input', () => {
    const input = [{ slug: 'a', y: 50 }, { slug: 'b', y: 50 }];
    spreadLabels(input, 14);
    expect(input.map((i) => i.y)).toEqual([50, 50]);
  });

  test('an empty set is fine', () => {
    expect(spreadLabels([], 14)).toEqual([]);
  });

  test('an overflowing stack slides back up into the plot', () => {
    // 90/95/100 with a 14px gap fan to 90/104/118; a 110px floor slides the
    // run up by 8 so the last label stays inside the viewBox instead of
    // clipping — the correction the Python renderer always had.
    const out = spreadLabels(
      [{ slug: 'a', y: 90 }, { slug: 'b', y: 95 }, { slug: 'c', y: 100 }],
      14, 10, 110,
    );
    expect(out.map((o) => o.y)).toEqual([82, 96, 110]);
  });
});

describe('table sorting', () => {
  test('numbers sort numerically, not as text', () => {
    // The bug this guards: "10" < "9" lexically, so a points column sorted as
    // text puts double digits in the wrong place the moment week 5 arrives.
    expect(compareValues('9', '10')).toBeLessThan(0);
    expect(compareValues('2.60', '1.45')).toBeGreaterThan(0);
    expect(compareValues('5', '5')).toBe(0);
  });

  test('non-numeric values fall back to text rather than being read as zero', () => {
    // "0-0" must not parse as 0 and silently tie with a real zero.
    expect(compareValues('3-1', '10-2')).toBeLessThan(0);
    expect(compareValues('Brian', 'Paul')).toBeLessThan(0);
  });

  test('an empty cell is not treated as zero', () => {
    expect(compareValues('', '5')).toBeLessThan(0);
    expect(compareValues('5', '')).toBeGreaterThan(0);
  });

  test('cellValue prefers an explicit sort key over the displayed text', () => {
    document.body.innerHTML =
      '<table><tbody><tr><td data-value="7">seven</td><td>plain</td></tr></tbody></table>';
    const row = document.querySelector('tr');
    expect(cellValue(row, 0)).toBe('7');
    expect(cellValue(row, 1)).toBe('plain');
    expect(cellValue(row, 9)).toBe('');
  });

  test('sortTable reorders the body in place', () => {
    document.body.innerHTML = `
      <table><tbody>
        <tr><td>b</td><td data-value="2">2</td></tr>
        <tr><td>a</td><td data-value="30">30</td></tr>
        <tr><td>c</td><td data-value="9">9</td></tr>
      </tbody></table>`;
    const table = document.querySelector('table');

    sortTable(table, 1, 1);
    expect([...table.tBodies[0].rows].map((r) => r.cells[1].textContent)).toEqual(
      ['2', '9', '30'],
    );

    sortTable(table, 1, -1);
    expect([...table.tBodies[0].rows].map((r) => r.cells[1].textContent)).toEqual(
      ['30', '9', '2'],
    );

    sortTable(table, 0, 1);
    expect([...table.tBodies[0].rows].map((r) => r.cells[0].textContent)).toEqual(
      ['a', 'b', 'c'],
    );
  });

  test('a table with no body is a no-op rather than a crash', () => {
    document.body.innerHTML = '<table><thead><tr><th>x</th></tr></thead></table>';
    expect(sortTable(document.querySelector('table'), 0, 1)).toEqual([]);
  });
});

describe('safeStorage', () => {
  test('reads and writes through to the underlying store', () => {
    const store = new Map();
    const s = safeStorage({
      getItem: (k) => store.get(k) ?? null,
      setItem: (k, v) => store.set(k, v),
    });
    s.set('a', '1');
    expect(s.get('a')).toBe('1');
    expect(s.get('missing')).toBeNull();
  });

  test('swallows errors when storage is unavailable', () => {
    // Private browsing throws on both read and write.
    const s = safeStorage({
      getItem() { throw new Error('denied'); },
      setItem() { throw new Error('denied'); },
    });
    expect(s.get('a')).toBeNull();
    expect(() => s.set('a', '1')).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// DOM wiring
// ---------------------------------------------------------------------------
function markup() {
  document.documentElement.removeAttribute('data-theme');
  document.body.innerHTML = `
    <button data-theme-toggle></button>
    <span class="row-points">39.20</span>
    <span class="hero-total">39.20</span>
    <time datetime="${ISO}" data-ts>${ISO}</time>
    <select data-tz-select>
      <option value="America/New_York">Eastern</option>
      <option value="America/Chicago">Central</option>
      <option value="UTC">UTC</option>
    </select>`;
}

function fakeWindow({ reduceMotion = false, prefersDark = true, store = new Map() } = {}) {
  const frames = [];
  return {
    localStorage: {
      getItem: (k) => store.get(k) ?? null,
      setItem: (k, v) => store.set(k, v),
    },
    matchMedia: (query) => ({
      matches: query.includes('reduced-motion') ? reduceMotion : prefersDark,
    }),
    performance: { now: () => 0 },
    requestAnimationFrame: (fn) => frames.push(fn),
    setTimeout: (fn) => fn(),
    _frames: frames,
    _store: store,
  };
}

describe('init', () => {
  beforeEach(markup);

  test('the toggle switches the theme and remembers it', () => {
    const win = fakeWindow({ prefersDark: true });
    init(document, win);

    document.querySelector('[data-theme-toggle]').click();
    expect(document.documentElement.dataset.theme).toBe('light');
    expect(win._store.get(THEME_KEY)).toBe('light');

    document.querySelector('[data-theme-toggle]').click();
    expect(document.documentElement.dataset.theme).toBe('dark');
  });

  test('timestamps localise to Eastern by default', () => {
    init(document, fakeWindow());
    const text = document.querySelector('time[data-ts]').textContent;
    expect(text).toContain('EST');
    expect(text).not.toBe(ISO);
    expect(document.querySelector('[data-tz-select]').value).toBe(DEFAULT_TZ);
  });

  test('changing the zone repaints every stamp and stores the choice', () => {
    const win = fakeWindow();
    init(document, win);

    const select = document.querySelector('[data-tz-select]');
    select.value = 'UTC';
    select.dispatchEvent(new window.Event('change'));

    expect(document.querySelector('time[data-ts]').textContent).toContain('UTC');
    expect(win._store.get(TZ_KEY)).toBe('UTC');
  });

  test('a stored zone is restored on the next visit', () => {
    const store = new Map([[TZ_KEY, 'America/Chicago']]);
    init(document, fakeWindow({ store }));
    expect(document.querySelector('[data-tz-select]').value).toBe('America/Chicago');
    expect(document.querySelector('time[data-ts]').textContent).toContain('CST');
  });

  test('a corrupt stored zone falls back to Eastern', () => {
    const store = new Map([[TZ_KEY, 'Mars/Olympus']]);
    init(document, fakeWindow({ store }));
    expect(document.querySelector('[data-tz-select]').value).toBe(DEFAULT_TZ);
  });

  test('a valid zone missing from the list still applies', () => {
    // The select honestly shows no choice, but the stamps render in the
    // stored zone — previously this snapped the viewer back to Eastern and
    // overwrote their stored preference with the emptied select.
    const store = new Map([[TZ_KEY, 'Europe/Berlin']]);
    init(document, fakeWindow({ store }));
    expect(document.querySelector('[data-tz-select]').value).toBe('');
    expect(store.get(TZ_KEY)).toBe('Europe/Berlin');
  });

  test('the odometer settles on the true value', () => {
    const win = fakeWindow();
    init(document, win);
    // The safety timeout fires synchronously in the fake window, so the real
    // number is on screen even if no animation frame ever runs.
    expect(document.querySelector('.row-points').textContent).toBe('39.20');
    expect(document.querySelector('.hero-total').textContent).toBe('39.20');
  });

  test('animation frames converge on the true value', () => {
    const win = fakeWindow();
    win.setTimeout = () => {}; // disable the safety net to test the frames
    let clock = 0;
    win.performance.now = () => clock;
    init(document, win);

    // Run the queued frames forward past the animation duration.
    for (let i = 0; i < 5 && win._frames.length; i += 1) {
      clock += 400;
      win._frames.shift()(clock);
    }
    expect(document.querySelector('.row-points').textContent).toBe('39.20');
  });

  test('reduced motion skips the count-up entirely', () => {
    const win = fakeWindow({ reduceMotion: true });
    init(document, win);
    expect(win._frames).toHaveLength(0);
    expect(document.querySelector('.row-points').textContent).toBe('39.20');
  });

  test('pages with no stamps or toggle do not throw', () => {
    document.body.innerHTML = '<p>nothing to wire up</p>';
    expect(() => init(document, fakeWindow())).not.toThrow();
  });

  test('stamps still localise without a picker', () => {
    document.body.innerHTML = `<time datetime="${ISO}" data-ts>${ISO}</time>`;
    init(document, fakeWindow());
    expect(document.querySelector('time[data-ts]').textContent).toContain('EST');
  });

  test('a stamp with no datetime attribute is left alone', () => {
    document.body.innerHTML = '<time data-ts>whenever</time>';
    init(document, fakeWindow());
    expect(document.querySelector('time[data-ts]').textContent).toBe('whenever');
  });

  test('non-numeric score text is not animated', () => {
    document.body.innerHTML = '<span class="row-points">—</span>';
    const win = fakeWindow();
    init(document, win);
    expect(document.querySelector('.row-points').textContent).toBe('—');
    expect(win._frames).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// "This is me"
// ---------------------------------------------------------------------------
function boardMarkup() {
  document.documentElement.removeAttribute('data-theme');
  document.documentElement.removeAttribute('data-me');
  document.body.innerHTML = `
    <a class="row" data-slug="brian-moore"><span class="badge you" data-me-only>you</span></a>
    <a class="row" data-slug="paul-moore"><span class="badge you" data-me-only>you</span></a>
    <select data-me-select>
      <option value="">Nobody in particular</option>
      <option value="brian-moore">Brian Moore</option>
      <option value="paul-moore">Paul Moore</option>
    </select>`;
}

describe('this is me', () => {
  beforeEach(boardMarkup);

  test('choosing a name marks that row and remembers it', () => {
    const win = fakeWindow();
    init(document, win);

    const select = document.querySelector('[data-me-select]');
    select.value = 'paul-moore';
    select.dispatchEvent(new window.Event('change'));

    expect(document.querySelector('[data-slug="paul-moore"]').classList.contains('is-me')).toBe(true);
    expect(document.querySelector('[data-slug="brian-moore"]').classList.contains('is-me')).toBe(false);
    expect(win._store.get(ME_KEY)).toBe('paul-moore');
    expect(document.documentElement.dataset.me).toBe('paul-moore');
  });

  test('a stored identity is restored on the next visit', () => {
    init(document, fakeWindow({ store: new Map([[ME_KEY, 'brian-moore']]) }));
    expect(document.querySelector('[data-me-select]').value).toBe('brian-moore');
    expect(document.querySelector('[data-slug="brian-moore"]').classList.contains('is-me')).toBe(true);
  });

  test('choosing nobody clears the highlight', () => {
    const win = fakeWindow({ store: new Map([[ME_KEY, 'brian-moore']]) });
    init(document, win);

    const select = document.querySelector('[data-me-select]');
    select.value = '';
    select.dispatchEvent(new window.Event('change'));

    expect(document.querySelectorAll('.is-me')).toHaveLength(0);
    expect(win._store.get(ME_KEY)).toBe('');
  });

  test('an identity that has left the pool is discarded, not left dangling', () => {
    // Next season's picks file will not contain last season's entrants, and a
    // stale value must not leave the picker showing a blank selection.
    const win = fakeWindow({ store: new Map([[ME_KEY, 'someone-who-left']]) });
    init(document, win);

    expect(document.querySelector('[data-me-select]').value).toBe('');
    expect(document.querySelectorAll('.is-me')).toHaveLength(0);
    expect(win._store.get(ME_KEY)).toBe('');
  });

  test('pages with no picker still apply a stored identity', () => {
    document.body.innerHTML = '<a class="row" data-slug="brian-moore"></a>';
    init(document, fakeWindow({ store: new Map([[ME_KEY, 'brian-moore']]) }));
    expect(document.querySelector('[data-slug="brian-moore"]').classList.contains('is-me')).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// This is me — multi mode
// ---------------------------------------------------------------------------
function multiBoardMarkup() {
  boardMarkup();
  // The chips container is what opts the picker into multi mode, and the
  // strip is where the pinned entries surface. Rows mirror the real board.
  document.body.innerHTML = `
    <div class="you-strip" data-you hidden></div>
    <div class="board">
      <a class="row" data-slug="brian-moore" href="/entrant/brian-moore/">
        <span class="row-rank">1</span><span class="row-name">Brian Moore</span>
        <span class="row-points">10.00</span>
        <span class="badge you" data-me-only>you</span>
      </a>
      <a class="row" data-slug="paul-moore" href="/entrant/paul-moore/">
        <span class="row-rank">2</span><span class="row-name">Paul Moore</span>
        <span class="row-points">8.00</span>
        <span class="badge you" data-me-only>you</span>
      </a>
    </div>
    <select data-me-select>
      <option value="">Nobody in particular</option>
      <option value="brian-moore">Brian Moore</option>
      <option value="paul-moore">Paul Moore</option>
    </select>
    <span class="me-chips" data-me-chips hidden></span>`;
}

describe('this is me, running two entries', () => {
  beforeEach(multiBoardMarkup);

  test('picking a second name pins it alongside the first', () => {
    const win = fakeWindow();
    init(document, win);
    const select = document.querySelector('[data-me-select]');

    select.value = 'brian-moore';
    select.dispatchEvent(new window.Event('change'));
    select.value = 'paul-moore';
    select.dispatchEvent(new window.Event('change'));

    expect(document.querySelectorAll('.row.is-me')).toHaveLength(2);
    expect(win._store.get(ME_KEY)).toBe('brian-moore paul-moore');
    expect(document.documentElement.dataset.me).toBe('brian-moore paul-moore');
    // The picker itself sits back on the placeholder; the chips carry state.
    expect(select.value).toBe('');
  });

  test('the strip shows one line per pinned entry, borrowing the board rows', () => {
    init(document, fakeWindow({ store: new Map([[ME_KEY, 'brian-moore paul-moore']]) }));
    const strip = document.querySelector('[data-you]');
    expect(strip.hidden).toBe(false);
    expect(strip.querySelectorAll('a')).toHaveLength(2);
    expect(strip.textContent).toContain('your entries');
    expect(strip.querySelector('a').getAttribute('href')).toBe('/entrant/brian-moore/');
  });

  test("a chip's own button unpins it and repaints everything", () => {
    const win = fakeWindow({ store: new Map([[ME_KEY, 'brian-moore paul-moore']]) });
    init(document, win);

    const chips = document.querySelectorAll('.me-chip');
    expect(chips).toHaveLength(2);
    chips[0].click();

    expect(win._store.get(ME_KEY)).toBe('paul-moore');
    expect(document.querySelectorAll('.row.is-me')).toHaveLength(1);
    expect(document.querySelectorAll('[data-you] a')).toHaveLength(1);
  });

  test('choosing nobody clears every pin, same as it always did', () => {
    const win = fakeWindow({ store: new Map([[ME_KEY, 'brian-moore paul-moore']]) });
    init(document, win);

    const select = document.querySelector('[data-me-select]');
    select.value = '';
    select.dispatchEvent(new window.Event('change'));

    expect(document.querySelectorAll('.is-me')).toHaveLength(0);
    expect(win._store.get(ME_KEY)).toBe('');
    expect(document.querySelector('[data-you]').hidden).toBe(true);
  });

  test('a departed entrant is dropped from a stored multi value', () => {
    const win = fakeWindow({ store: new Map([[ME_KEY, 'brian-moore someone-who-left']]) });
    init(document, win);
    expect(win._store.get(ME_KEY)).toBe('brian-moore');
    expect(document.querySelectorAll('.row.is-me')).toHaveLength(1);
  });

  test('the pin count is capped', () => {
    expect(ME_LIMIT).toBeGreaterThan(1);
  });
});

// ---------------------------------------------------------------------------
// Comparison chart
// ---------------------------------------------------------------------------
function compareMarkup(slugs = ['brian-moore', 'paul-moore', 'brenda-moore']) {
  document.documentElement.removeAttribute('data-me');
  const picks = slugs
    .map((s) => `<button class="pick" data-pick="${s}" aria-pressed="false">${s}</button>`)
    .join('');
  const lines = slugs
    .map(
      (s, i) =>
        `<polyline class="cmp-line" data-entrant="${s}"></polyline>` +
        `<circle class="cmp-dot" data-entrant="${s}"></circle>` +
        `<text class="cmp-label" data-entrant="${s}" data-y="${40 + i * 2}"></text>`,
    )
    .join('');
  document.body.innerHTML = `
    <div class="compare-picks">${picks}</div>
    <p data-compare-empty>nothing picked</p>
    <div data-compare><svg>${lines}</svg></div>`;
}

const pickedSlugs = () =>
  [...document.querySelectorAll('.cmp-line.is-picked')].map((el) => el.dataset.entrant);

describe('comparison chart', () => {
  beforeEach(() => compareMarkup());

  test('nothing is picked by default and the prompt is visible', () => {
    init(document, fakeWindow());
    expect(pickedSlugs()).toEqual([]);
    expect(document.querySelector('[data-compare-empty]').hidden).toBe(false);
  });

  test('picking someone lights their line, dot and label together', () => {
    const win = fakeWindow();
    init(document, win);
    document.querySelector('[data-pick="paul-moore"]').click();

    expect(pickedSlugs()).toEqual(['paul-moore']);
    for (const cls of ['.cmp-line', '.cmp-dot', '.cmp-label']) {
      const el = document.querySelector(`${cls}[data-entrant="paul-moore"]`);
      expect(el.classList.contains('is-picked')).toBe(true);
      expect(el.style.getPropertyValue('--pick-color')).toBe(PICK_COLORS[0]);
    }
    expect(document.querySelector('[data-pick="paul-moore"]').getAttribute('aria-pressed')).toBe('true');
    expect(document.querySelector('[data-compare-empty]').hidden).toBe(true);
    expect(win._store.get(COMPARE_KEY)).toBe('paul-moore');
  });

  test('colours follow pick order, so the first pick is always the first colour', () => {
    init(document, fakeWindow());
    document.querySelector('[data-pick="brenda-moore"]').click();
    document.querySelector('[data-pick="brian-moore"]').click();

    const color = (slug) =>
      document.querySelector(`.cmp-line[data-entrant="${slug}"]`).style.getPropertyValue('--pick-color');
    expect(color('brenda-moore')).toBe(PICK_COLORS[0]);
    expect(color('brian-moore')).toBe(PICK_COLORS[1]);
  });

  test('clicking again unpicks and clears the colour', () => {
    init(document, fakeWindow());
    const button = document.querySelector('[data-pick="brian-moore"]');
    button.click();
    button.click();

    expect(pickedSlugs()).toEqual([]);
    expect(button.getAttribute('aria-pressed')).toBe('false');
    const line = document.querySelector('.cmp-line[data-entrant="brian-moore"]');
    expect(line.style.getPropertyValue('--pick-color')).toBe('');
  });

  test('past the palette the oldest pick drops rather than reusing a colour', () => {
    const many = Array.from({ length: PICK_COLORS.length + 1 }, (_, i) => `p${i}`);
    compareMarkup(many);
    init(document, fakeWindow());
    for (const slug of many) document.querySelector(`[data-pick="${slug}"]`).click();

    const picked = pickedSlugs();
    expect(picked).toHaveLength(PICK_COLORS.length);
    expect(picked).not.toContain('p0'); // the first one picked made way
    expect(new Set(picked.map((s) =>
      document.querySelector(`.cmp-line[data-entrant="${s}"]`).style.getPropertyValue('--pick-color'),
    )).size).toBe(PICK_COLORS.length);
  });

  test('it opens on your own line once you have said who you are', () => {
    // The whole chain: the stored identity is applied by initMe, and initCompare
    // reads it from the document. Order matters here — initMe writes the
    // attribute that initCompare then reads — so this covers the wiring too.
    init(document, fakeWindow({ store: new Map([[ME_KEY, 'brenda-moore']]) }));
    expect(pickedSlugs()).toEqual(['brenda-moore']);
  });

  test('a stored selection beats the identity default', () => {
    const store = new Map([
      [ME_KEY, 'brenda-moore'],
      [COMPARE_KEY, 'brian-moore paul-moore'],
    ]);
    init(document, fakeWindow({ store }));
    expect(pickedSlugs().sort()).toEqual(['brian-moore', 'paul-moore']);
  });

  test('stored names that are no longer in the pool are dropped', () => {
    init(document, fakeWindow({ store: new Map([[COMPARE_KEY, 'ghost brian-moore']]) }));
    expect(pickedSlugs()).toEqual(['brian-moore']);
  });

  test('colliding endpoint labels are pushed apart, dots stay put', () => {
    document.body.innerHTML = `
      <div class="compare-picks">
        <button class="pick" data-pick="a" aria-pressed="false">a</button>
        <button class="pick" data-pick="b" aria-pressed="false">b</button>
      </div>
      <div data-compare><svg>
        <polyline class="cmp-line" data-entrant="a"></polyline>
        <text class="cmp-label" data-entrant="a" data-y="50"></text>
        <polyline class="cmp-line" data-entrant="b"></polyline>
        <text class="cmp-label" data-entrant="b" data-y="50"></text>
      </svg></div>`;
    init(document, fakeWindow());
    document.querySelector('[data-pick="a"]').click();
    document.querySelector('[data-pick="b"]').click();

    const ys = [...document.querySelectorAll('.cmp-label')].map((el) => Number(el.getAttribute('y')));
    expect(new Set(ys).size).toBe(2);
    expect(Math.abs(ys[0] - ys[1])).toBeGreaterThanOrEqual(14);
  });

  test('a page with no chart does not throw', () => {
    document.body.innerHTML = '<p>no chart here</p>';
    expect(() => init(document, fakeWindow())).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// Sortable tables
// ---------------------------------------------------------------------------
describe('sortable table headers', () => {
  beforeEach(() => {
    document.body.innerHTML = `
      <table>
        <thead><tr><th data-sort>Team</th><th data-sort>Points</th></tr></thead>
        <tbody>
          <tr><th data-value="ARI">ARI</th><td data-value="4">4.00</td></tr>
          <tr><th data-value="KC">KC</th><td data-value="30">30.00</td></tr>
          <tr><th data-value="DAL">DAL</th><td data-value="12">12.00</td></tr>
        </tbody>
      </table>`;
  });

  const column = (i) => [...document.querySelectorAll('tbody tr')].map((r) => r.children[i].textContent);

  test('clicking a heading sorts by it and marks the direction', () => {
    init(document, fakeWindow());
    const th = document.querySelectorAll('th[data-sort]')[1];
    th.click();

    expect(column(1)).toEqual(['4.00', '12.00', '30.00']);
    expect(th.getAttribute('aria-sort')).toBe('ascending');
  });

  test('clicking the same heading again reverses it', () => {
    init(document, fakeWindow());
    const th = document.querySelectorAll('th[data-sort]')[1];
    th.click();
    th.click();

    expect(column(1)).toEqual(['30.00', '12.00', '4.00']);
    expect(th.getAttribute('aria-sort')).toBe('descending');
  });

  test('a numeric column opens biggest-first', () => {
    // Templates mark number columns with class="num"; those lead with the
    // biggest value — a points column is useless smallest-first.
    document.querySelectorAll('th[data-sort]')[1].classList.add('num');
    init(document, fakeWindow());
    const th = document.querySelectorAll('th[data-sort]')[1];
    th.click();

    expect(column(1)).toEqual(['30.00', '12.00', '4.00']);
    expect(th.getAttribute('aria-sort')).toBe('descending');

    th.click();
    expect(column(1)).toEqual(['4.00', '12.00', '30.00']);
    expect(th.getAttribute('aria-sort')).toBe('ascending');
  });

  test('sorting a different column clears the previous marker', () => {
    init(document, fakeWindow());
    const [first, second] = document.querySelectorAll('th[data-sort]');
    second.click();
    first.click();

    expect(column(0)).toEqual(['ARI', 'DAL', 'KC']);
    expect(second.hasAttribute('aria-sort')).toBe(false);
    expect(first.getAttribute('aria-sort')).toBe('ascending');
  });

  test('headings are reachable and operable from the keyboard', () => {
    init(document, fakeWindow());
    const th = document.querySelectorAll('th[data-sort]')[1];
    expect(th.tabIndex).toBe(0);

    th.dispatchEvent(new window.KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    expect(column(1)).toEqual(['4.00', '12.00', '30.00']);

    th.dispatchEvent(new window.KeyboardEvent('keydown', { key: ' ', bubbles: true }));
    expect(column(1)).toEqual(['30.00', '12.00', '4.00']);
  });

  test('other keys are ignored', () => {
    init(document, fakeWindow());
    const th = document.querySelectorAll('th[data-sort]')[1];
    th.dispatchEvent(new window.KeyboardEvent('keydown', { key: 'a', bubbles: true }));
    expect(column(1)).toEqual(['4.00', '30.00', '12.00']);
  });

  test('a table with no sortable headings is left alone', () => {
    document.body.innerHTML = '<table><thead><tr><th>x</th></tr></thead><tbody><tr><td>1</td></tr></tbody></table>';
    expect(() => init(document, fakeWindow())).not.toThrow();
    expect(document.querySelector('th').tabIndex).toBe(-1);
  });
});
