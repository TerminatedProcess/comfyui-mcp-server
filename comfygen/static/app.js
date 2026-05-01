/* ComfyGen frontend */

const API = '';
let selected = new Set();
let galleryItems = [];
let activeLoras = [];
let currentFamily = 'ALL';
let generating = false;
let viewerPath = null;

// ── API helpers ──

async function api(url, opts) {
    const res = await fetch(API + url, opts);
    return res.json();
}

async function apiPost(url, data) {
    return api(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
    });
}

// ── Init ──

async function init() {
    await loadSamplers();
    await loadFamilies();
    await refresh();
    loadGallery();
}

async function refresh() {
    const status = await api('/api/status');
    const dot = document.getElementById('status-dot');
    const text = document.getElementById('status-text');
    dot.className = 'dot' + (status.online ? ' online' : '');
    text.textContent = status.online
        ? `${status.model_count} models loaded`
        : 'ComfyUI offline';

    if (status.online) {
        await loadModels();
        await loadLoras();
        await loadClips();
        await loadVaes();
    }
}

// ── Dropdowns ──

function populateSelect(id, items, defaultVal) {
    const sel = document.getElementById(id);
    const prev = sel.value;
    sel.innerHTML = '';
    for (const item of items) {
        const opt = document.createElement('option');
        opt.value = item;
        opt.textContent = item;
        sel.appendChild(opt);
    }
    if (defaultVal && items.includes(defaultVal)) {
        sel.value = defaultVal;
    } else if (prev && items.includes(prev)) {
        sel.value = prev;
    }
}

async function loadModels() {
    const data = await api(`/api/models?family=${currentFamily}`);
    populateSelect('model', data.models);
}

async function loadLoras() {
    const data = await api(`/api/loras?family=${currentFamily}`);
    const sel = document.getElementById('lora-select');
    sel.innerHTML = '<option value="">Select LoRA...</option>';
    for (const l of data.loras) {
        const opt = document.createElement('option');
        opt.value = l;
        opt.textContent = l;
        sel.appendChild(opt);
    }
}

async function loadClips() {
    const data = await api('/api/clips');
    populateSelect('clip', data.clips, data.default);
}

async function loadVaes() {
    const data = await api('/api/vaes');
    populateSelect('vae', data.vaes, data.default);
}

async function loadSamplers() {
    const data = await api('/api/samplers');
    populateSelect('sampler', data.samplers, 'res_multistep');
    populateSelect('scheduler', data.schedulers, 'simple');
}

async function loadFamilies() {
    const data = await api('/api/families');
    const container = document.getElementById('family-pills');
    container.innerHTML = '';
    for (const f of data.families) {
        const pill = document.createElement('span');
        pill.className = 'pill' + (f === currentFamily ? ' active' : '');
        pill.textContent = f;
        pill.onclick = () => selectFamily(f);
        container.appendChild(pill);
    }
}

async function selectFamily(family) {
    currentFamily = family;
    document.querySelectorAll('.pill').forEach(p => {
        p.classList.toggle('active', p.textContent === family);
    });
    const data = await api(`/api/models?family=${family}`);
    populateSelect('model', data.models);
    if (data.preset) {
        if (data.preset.steps) document.getElementById('steps').value = data.preset.steps;
        if (data.preset.cfg !== undefined) document.getElementById('cfg').value = data.preset.cfg;
        if (data.preset.sampler) document.getElementById('sampler').value = data.preset.sampler;
        if (data.preset.scheduler) document.getElementById('scheduler').value = data.preset.scheduler;
    }
    await loadLoras();
}

// ── LoRA management ──

function addLora() {
    const sel = document.getElementById('lora-select');
    const weight = parseFloat(document.getElementById('lora-weight').value) || 0.75;
    if (!sel.value) return;
    if (activeLoras.find(l => l.name === sel.value)) return;
    activeLoras.push({ name: sel.value, weight });
    sel.value = '';
    renderLoras();
}

function removeLora(index) {
    activeLoras.splice(index, 1);
    renderLoras();
}

function renderLoras() {
    const container = document.getElementById('lora-list');
    if (!activeLoras.length) {
        container.innerHTML = '';
        return;
    }
    container.innerHTML = activeLoras.map((l, i) =>
        `<div class="lora-item">
            <span class="name" title="${l.name}">${l.name}</span>
            <span class="weight">${l.weight}</span>
            <span class="remove" onclick="removeLora(${i})">×</span>
        </div>`
    ).join('');
}

function getLoraText() {
    return activeLoras.map(l => `${l.name}:${l.weight}`).join('\n');
}

// ── Generate ──

async function generate() {
    if (generating) return;
    generating = true;
    const btn = document.getElementById('generate-btn');
    btn.disabled = true;
    btn.textContent = 'Generating...';

    const viewer = document.getElementById('viewer');
    const overlay = document.createElement('div');
    overlay.className = 'generating-overlay';
    overlay.innerHTML = '<div class="spinner"></div>';
    viewer.parentElement.appendChild(overlay);

    try {
        const data = await apiPost('/api/generate', {
            prompt: document.getElementById('prompt').value,
            neg_prompt: document.getElementById('neg-prompt').value,
            width: parseInt(document.getElementById('width').value),
            height: parseInt(document.getElementById('height').value),
            steps: parseInt(document.getElementById('steps').value),
            cfg: parseFloat(document.getElementById('cfg').value),
            seed: parseInt(document.getElementById('seed').value),
            random_seed: document.getElementById('random-seed').checked,
            sampler: document.getElementById('sampler').value,
            scheduler: document.getElementById('scheduler').value,
            model: document.getElementById('model').value,
            clip: document.getElementById('clip').value,
            vae: document.getElementById('vae').value,
            loras: getLoraText(),
        });

        if (data.error) {
            setStatus(data.error);
        } else {
            showInViewer(data.path, false);
            setStatus(`${data.filename} | seed: ${data.seed}`);
            loadGallery();
        }
    } catch (e) {
        setStatus('Generation failed: ' + e.message);
    } finally {
        overlay.remove();
        generating = false;
        btn.disabled = false;
        btn.textContent = 'Generate';
    }
}

// ── Viewer ──

function showInViewer(path, isVideo) {
    viewerPath = path;
    const viewer = document.getElementById('viewer');
    const info = document.getElementById('viewer-info');
    const mediaUrl = `/api/media?path=${encodeURIComponent(path)}`;

    if (isVideo) {
        viewer.innerHTML = `<video src="${mediaUrl}" controls autoplay loop style="max-width:100%;max-height:100%"></video>`;
    } else {
        viewer.innerHTML = `<img src="${mediaUrl}" onclick="openFullscreen('${mediaUrl}', false)">`;
    }
    info.style.display = 'block';
    info.textContent = path.split('/').pop();
}

function openFullscreen(url, isVideo) {
    const fs = document.getElementById('fullscreen');
    if (isVideo) {
        fs.innerHTML = `<video src="${url}" controls autoplay loop></video>`;
    } else {
        fs.innerHTML = `<img src="${url}">`;
    }
    fs.style.display = 'flex';
}

function closeFullscreen() {
    document.getElementById('fullscreen').style.display = 'none';
}

// ── Gallery ──

async function loadGallery() {
    const data = await api('/api/gallery');
    galleryItems = data.items;
    renderGallery();
}

function renderGallery() {
    const grid = document.getElementById('gallery');
    if (!galleryItems.length) {
        grid.innerHTML = '<div class="gallery-empty">No images yet</div>';
        return;
    }
    grid.innerHTML = galleryItems.map(item => {
        const url = `/api/media?path=${encodeURIComponent(item.path)}`;
        const sel = selected.has(item.path) ? ' selected' : '';
        const media = item.is_video
            ? `<video src="${url}" muted preload="metadata"></video><span class="video-icon">▶</span>`
            : `<img src="${url}" loading="lazy">`;
        return `<div class="gallery-item${sel}" data-path="${item.path}" data-video="${item.is_video}">
            ${media}<div class="badge"></div>
        </div>`;
    }).join('');
    updateSelCount();
}

function updateSelCount() {
    const el = document.getElementById('sel-count');
    el.textContent = selected.size > 0 ? `${selected.size} selected` : '';
}

// ── Gallery interactions ──

document.addEventListener('click', e => {
    const item = e.target.closest('.gallery-item');
    if (!item) return;
    e.preventDefault();
    const path = item.dataset.path;
    const isVideo = item.dataset.video === 'true';

    if (e.ctrlKey || e.metaKey) {
        if (selected.has(path)) {
            selected.delete(path);
            item.classList.remove('selected');
        } else {
            selected.add(path);
            item.classList.add('selected');
        }
    } else {
        document.querySelectorAll('.gallery-item.selected').forEach(i => i.classList.remove('selected'));
        selected.clear();
        selected.add(path);
        item.classList.add('selected');
    }
    updateSelCount();
    showInViewer(path, isVideo);
});

document.addEventListener('dblclick', e => {
    const item = e.target.closest('.gallery-item');
    if (!item) return;
    e.preventDefault();
    loadSettingsFromImage(item.dataset.path);
});

// ── Keyboard shortcuts ──

document.addEventListener('keydown', e => {
    // Ctrl+Enter → generate
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        generate();
        return;
    }

    // Escape → close fullscreen
    if (e.key === 'Escape') {
        closeFullscreen();
        return;
    }

    // Gallery-focused shortcuts
    const gallery = document.getElementById('gallery');
    const inGallery = gallery.contains(document.activeElement) || document.activeElement === gallery;
    const inInput = ['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement.tagName);

    // Ctrl+A → select all (only when gallery focused or no input focused)
    if ((e.ctrlKey || e.metaKey) && e.key === 'a' && !inInput) {
        e.preventDefault();
        selected.clear();
        document.querySelectorAll('.gallery-item').forEach(item => {
            selected.add(item.dataset.path);
            item.classList.add('selected');
        });
        updateSelCount();
        return;
    }

    // Delete → delete selected (only when not in input)
    if (e.key === 'Delete' && !inInput && selected.size > 0) {
        e.preventDefault();
        deleteSelected();
        return;
    }
});

// ── Actions ──

async function deleteSelected() {
    if (selected.size === 0) return;
    const paths = Array.from(selected);
    const data = await apiPost('/api/gallery/delete', { paths });
    setStatus(`Deleted ${data.deleted} file(s)`);
    selected.clear();
    viewerPath = null;
    document.getElementById('viewer').innerHTML = '<div class="viewer-empty">Generate an image or select from gallery</div>';
    document.getElementById('viewer-info').style.display = 'none';
    loadGallery();
}

async function loadSettingsFromImage(path) {
    const data = await apiPost('/api/gallery/load-settings', { path });
    if (data.error) {
        setStatus(data.error);
        return;
    }
    document.getElementById('prompt').value = data.prompt || '';
    document.getElementById('neg-prompt').value = data.neg_prompt || '';
    document.getElementById('width').value = data.width || 1024;
    document.getElementById('height').value = data.height || 1024;
    document.getElementById('steps').value = data.steps || 20;
    document.getElementById('cfg').value = data.cfg || 7;
    document.getElementById('seed').value = data.seed || 42;
    document.getElementById('random-seed').checked = false;
    if (data.sampler) document.getElementById('sampler').value = data.sampler;
    if (data.scheduler) document.getElementById('scheduler').value = data.scheduler;
    if (data.model) {
        const modelSel = document.getElementById('model');
        if ([...modelSel.options].some(o => o.value === data.model)) {
            modelSel.value = data.model;
        }
    }
    if (data.loras) {
        activeLoras = data.loras.split('\n').filter(l => l.trim()).map(l => {
            const parts = l.rsplit ? l.split(':') : l.split(':');
            const weight = parseFloat(parts.pop()) || 1.0;
            const name = parts.join(':');
            return { name, weight };
        });
    } else {
        activeLoras = [];
    }
    renderLoras();
    setStatus(`Loaded settings: ${data.filename}`);
}

async function pullFromInvoke() {
    setStatus('Pulling from InvokeAI...');
    const data = await apiPost('/api/pull-invoke', {});
    if (data.error) {
        setStatus(data.error);
        return;
    }
    document.getElementById('prompt').value = data.prompt || '';
    document.getElementById('neg-prompt').value = data.neg_prompt || '';
    document.getElementById('width').value = data.width || 1024;
    document.getElementById('height').value = data.height || 1024;
    document.getElementById('steps').value = data.steps || 20;
    document.getElementById('cfg').value = data.cfg || 7;
    document.getElementById('seed').value = data.seed || 42;
    document.getElementById('random-seed').checked = data.random_seed !== false;
    if (data.sampler) document.getElementById('sampler').value = data.sampler;
    if (data.scheduler) document.getElementById('scheduler').value = data.scheduler;
    if (data.model) {
        const modelSel = document.getElementById('model');
        // If model not in list, refresh models first
        if (![...modelSel.options].some(o => o.value === data.model)) {
            await refresh();
        }
        modelSel.value = data.model;
    }
    if (data.loras) {
        activeLoras = data.loras.split('\n').filter(l => l.trim()).map(l => {
            const parts = l.split(':');
            const weight = parseFloat(parts.pop()) || 1.0;
            const name = parts.join(':');
            return { name, weight };
        });
    } else {
        activeLoras = [];
    }
    renderLoras();
    setStatus(data.status || 'Pulled from InvokeAI');
}

function setStatus(msg) {
    document.getElementById('status-text').textContent = msg;
}

// ── Start ──

init();
