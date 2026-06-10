const CACHE       = 'invader-hunter-v7';
const SHARE_CACHE = 'invader-share-v1';

const PRECACHE = [
  './manifest.json',
  './icon-192.png',
  './icon-512.png',
  'https://fonts.googleapis.com/css2?family=Press+Start+2P&family=Share+Tech+Mono&display=swap',
  'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css',
  'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js',
  'https://unpkg.com/leaflet.markercluster@1.5.3/dist/leaflet.markercluster.js',
  'https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.css',
  'https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.Default.css',
];

// Install: pre-cache app shell (not index.html — it's network-first)
self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE)
      .then(cache => cache.addAll(PRECACHE))
      .then(() => self.skipWaiting())
  );
});

// Activate: clean up old caches
self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE && k !== SHARE_CACHE).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

// Fetch
self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);

  // Share Target: intercept POST from Android share sheet
  if (url.pathname.endsWith('/share-target') && e.request.method === 'POST') {
    e.respondWith((async () => {
      const data = await e.request.formData();
      const file = data.get('file');
      if (file) {
        const text = await file.text();
        const cache = await caches.open(SHARE_CACHE);
        await cache.put('./shared-import', new Response(text, {
          headers: { 'Content-Type': 'text/plain' }
        }));
      }
      return Response.redirect('./index.html?from=share', 303);
    })());
    return;
  }

  // Map tiles: always network, no caching
  if (url.hostname.includes('cartocdn.com')) {
    e.respondWith(fetch(e.request).catch(() => new Response('', { status: 503 })));
    return;
  }

  // index.html: network-first, fall back to cache when offline
  if (url.pathname.endsWith('/') || url.pathname.endsWith('index.html')) {
    e.respondWith(
      fetch(e.request)
        .then(response => {
          const clone = response.clone();
          caches.open(CACHE).then(cache => cache.put(e.request, clone));
          return response;
        })
        .catch(() => caches.match(e.request))
    );
    return;
  }

  // Everything else: cache-first (icons, fonts, Leaflet, model, embeddings…)
  e.respondWith(
    caches.match(e.request).then(cached => {
      if (cached) return cached;
      return fetch(e.request).then(response => {
        if (e.request.method === 'GET' && response.status === 200) {
          const clone = response.clone();
          caches.open(CACHE).then(cache => cache.put(e.request, clone));
        }
        return response;
      });
    })
  );
});
