import test from 'node:test';
import assert from 'node:assert/strict';
import { canonicalViewHash, mountViewRouter, resolveSiteView, SITE_VIEWS } from '../src/view-router.js';

test('existing hashes resolve to stable site views', () => {
  const cases = {
    '#top': 'compute', '#compute': 'compute',
    '#route': 'execution',
    '#capabilities': 'ecosystem', '#phases': 'ecosystem',
    '#trust': 'measurements', '#benchmarks': 'measurements', '#provenance': 'measurements',
    '#methods': 'methods', '#morton': 'methods', '#pruning': 'methods',
    '#quickstart': 'install',
  };
  for (const [hash, view] of Object.entries(cases)) assert.equal(resolveSiteView(hash), view, hash);
  assert.equal(resolveSiteView('#unknown'), 'compute');
  assert.equal(resolveSiteView('#UNKNOWN', 'install'), 'install');
  assert.equal(SITE_VIEWS.length, 6);
  assert.equal(canonicalViewHash('measurements'), '#benchmarks');
  assert.throws(() => canonicalViewHash('nothing'), RangeError);
});

class MockElement {
  constructor(attributes = {}) {
    this.attributes = new Map(Object.entries(attributes));
    this.hidden = false;
    this.scrollCount = 0;
  }
  hasAttribute(name) { return this.attributes.has(name); }
  getAttribute(name) { return this.attributes.get(name) ?? null; }
  setAttribute(name, value) { this.attributes.set(name, String(value)); }
  removeAttribute(name) { this.attributes.delete(name); }
  toggleAttribute(name, force) {
    if (force) this.setAttribute(name, '');
    else this.removeAttribute(name);
  }
  closest(selector) { return selector === '[hidden]' && this.hidden ? this : null; }
  scrollIntoView() { this.scrollCount += 1; }
}

function fixture(hash = '#top') {
  const sectionIds = [
    ['top', 'compute'], ['route', 'execution'], ['capabilities', 'ecosystem'],
    ['phases', 'ecosystem'], ['trust', 'measurements'], ['benchmarks', 'measurements'],
    ['provenance', 'measurements'], ['methods', 'methods'], ['morton', 'methods'],
    ['pruning', 'methods'], ['quickstart', 'install'],
  ];
  const sections = sectionIds.map(([id, view]) => new MockElement({ id, 'data-site-view': view }));
  const byId = new Map(sections.map(node => [node.getAttribute('id'), node]));
  const main = new MockElement();
  main.children = sections;
  const skipLink = new MockElement({ href: '#content', class: 'skip-link' });
  main.ownerDocument = {
    getElementById: id => byId.get(id) ?? null,
    querySelector: selector => selector === '.skip-link' ? skipLink : null,
  };
  const links = ['#compute', '#route', '#capabilities', '#methods', '#benchmarks', '#quickstart']
    .map(href => new MockElement({ href }));
  const nav = { querySelectorAll: () => links };
  const listeners = new Map();
  const windowRef = {
    location: { hash },
    addEventListener(type, listener) { listeners.set(type, listener); },
    removeEventListener(type) { listeners.delete(type); },
    requestAnimationFrame(callback) { callback(); },
    dispatchHashChange() { listeners.get('hashchange')?.(); },
  };
  return { main, sections, nav, links, skipLink, windowRef, byId, listeners };
}

function visibleIds(sections) {
  return sections.filter(section => !section.hidden).map(section => section.getAttribute('id'));
}

test('hash changes switch views, preserve grouped sections, and mark one nav item', () => {
  const f = fixture('#top');
  const router = mountViewRouter({ main: f.main, nav: f.nav, windowRef: f.windowRef });
  assert.deepEqual(visibleIds(f.sections), ['top']);
  assert.equal(f.main.getAttribute('data-current-view'), 'compute');
  assert.equal(f.links[0].getAttribute('aria-current'), 'page');

  f.windowRef.location.hash = '#provenance';
  f.windowRef.dispatchHashChange();
  assert.equal(router.currentView, 'measurements');
  assert.deepEqual(visibleIds(f.sections), ['trust', 'benchmarks', 'provenance']);
  assert.equal(f.links[4].getAttribute('aria-current'), 'page');
  assert.equal(f.links.filter(link => link.hasAttribute('aria-current')).length, 1);
  assert.equal(f.byId.get('trust').scrollCount, 1);

  f.windowRef.location.hash = '#pruning';
  f.windowRef.dispatchHashChange();
  assert.deepEqual(visibleIds(f.sections), ['methods', 'morton', 'pruning']);
  assert.equal(f.byId.get('methods').scrollCount, 1);

  f.windowRef.location.hash = '#route';
  f.windowRef.dispatchHashChange();
  assert.deepEqual(visibleIds(f.sections), ['route']);
  assert.equal(f.links[1].getAttribute('aria-current'), 'page');

  router.destroy();
  assert.equal(f.listeners.has('hashchange'), false);
  assert.equal(f.main.hasAttribute('data-current-view'), false);
  assert.equal(f.sections.every(section => !section.hidden), true);
  assert.equal(f.links.some(link => link.hasAttribute('aria-current')), false);
});

test('direct deep links, browser history, programmatic navigation, and unknown hashes work', () => {
  const f = fixture('#quickstart');
  const router = mountViewRouter({ main: f.main, nav: f.nav, windowRef: f.windowRef });
  assert.deepEqual(visibleIds(f.sections), ['quickstart']);
  assert.equal(f.byId.get('quickstart').scrollCount, 1);

  assert.equal(router.navigate('ecosystem'), 'ecosystem');
  assert.equal(f.windowRef.location.hash, '#capabilities');
  assert.deepEqual(visibleIds(f.sections), ['capabilities', 'phases']);

  // A back/forward hashchange uses the restored URL, not stale router state.
  f.windowRef.location.hash = '#quickstart';
  f.windowRef.dispatchHashChange();
  assert.equal(router.currentView, 'install');
  f.windowRef.location.hash = '#missing';
  f.windowRef.dispatchHashChange();
  assert.equal(router.currentView, 'compute');
  assert.deepEqual(visibleIds(f.sections), ['top']);
  assert.equal(f.byId.get('top').scrollCount, 1);
  router.destroy();
});

test('skip link follows the visible page and view changes notify canvas owners', () => {
  const f = fixture('#morton');
  const transitions = [];
  const router = mountViewRouter({
    main: f.main, nav: f.nav, windowRef: f.windowRef,
    onViewChange: transition => {
      assert.deepEqual(visibleIds(f.sections), transition.view === 'methods'
        ? ['methods', 'morton', 'pruning'] : ['trust', 'benchmarks', 'provenance']);
      transitions.push(transition);
    },
  });
  assert.equal(f.skipLink.getAttribute('href'), '#methods');
  assert.deepEqual(transitions, [{ view: 'methods', previousView: null }]);

  f.windowRef.location.hash = '#pruning';
  f.windowRef.dispatchHashChange();
  assert.equal(transitions.length, 1, 'same-view subanchor needs no canvas resize notification');
  assert.equal(f.skipLink.getAttribute('href'), '#methods');
  assert.equal(f.byId.get('methods').scrollCount, 2, 'Method subanchors retain the shared chooser');

  f.windowRef.location.hash = '#provenance';
  f.windowRef.dispatchHashChange();
  assert.equal(f.skipLink.getAttribute('href'), '#benchmarks');
  assert.deepEqual(transitions[1], { view: 'measurements', previousView: 'methods' });
  router.destroy();
  assert.equal(f.skipLink.getAttribute('href'), '#content');
});

test('invalid configurations fail clearly', () => {
  const f = fixture();
  assert.throws(() => mountViewRouter({ main: null, nav: f.nav, windowRef: f.windowRef }), TypeError);
  assert.throws(() => mountViewRouter({ main: f.main, nav: f.nav, defaultView: 'other', windowRef: f.windowRef }), RangeError);
  assert.throws(() => mountViewRouter({ main: f.main, nav: f.nav, onViewChange: true, windowRef: f.windowRef }), TypeError);
  f.sections[0].setAttribute('data-site-view', 'other');
  assert.throws(() => mountViewRouter({ main: f.main, nav: f.nav, windowRef: f.windowRef }), RangeError);
});
