let currentJobId = null;
let currentJobUrl = '';
let pollTimer = null;

const $ = (id) => document.getElementById(id);
const message = (text) => { $('message').textContent = text; };

function setActions(enabled) {
  $('copy').disabled = !enabled;
  $('post-x').disabled = !enabled;
  $('delete').disabled = !currentJobId;
}

function scriptLines(job) {
  const script = job.kurage_script || job.script || {};
  const scenes = Array.isArray(script.scenes) ? script.scenes : [];
  if (scenes.length) return scenes.map((s) => s.narration || '').filter(Boolean);
  const outline = Array.isArray(job.script_outline) ? job.script_outline : [];
  return outline;
}

function jobTitle(job) {
  return job.kurage_title || job.title || job.source_title || job.url || '生成中';
}

function escapeHtml(text) {
  return String(text || '').replace(/[&<>"']/g, (ch) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
  }[ch]));
}

function statusLabel(job) {
  const labels = {
    queued: '待機中',
    analyzing: 'URL解析中',
    downloading: '元動画取得中',
    transcribing: '文字起こし中',
    planning: '台本生成中',
    generating: 'Kurage動画生成中',
    done: '完了',
    error: 'エラー'
  };
  return labels[job.status] || job.status || '不明';
}

function renderJob(job) {
  currentJobId = job.id;
  currentJobUrl = job.url || '';
  if (job.url) $('source-url').value = job.url;
  $('status').textContent = `${job.status || 'unknown'} ${job.progress ?? 0}%`;
  $('title').textContent = jobTitle(job);
  $('summary').textContent = job.summary || job.reference_analysis?.core_claim || job.analysis?.reference_analysis?.core_claim || job.analysis?.summary || job.transcript_preview || '解析中です。';
  const list = $('script');
  list.innerHTML = '';
  for (const line of scriptLines(job)) {
    const li = document.createElement('li');
    li.textContent = line;
    list.appendChild(li);
  }
  const link = job.video_url || job.kurage_url || '#';
  $('kurage-link').href = link;
  const player = $('player-wrap');
  if (job.status === 'done' && job.kurage_job_id) {
    player.className = 'player';
    const videoUrl = `https://kurage.exbridge.jp/kuragev.php?proxy=video&job_id=${encodeURIComponent(job.kurage_job_id)}`;
    player.innerHTML = `<video src="${videoUrl}" controls playsinline preload="metadata"></video>`;
    setActions(true);
  } else if (job.status === 'error') {
    player.className = 'player empty';
    player.innerHTML = `<span>エラー: ${(job.error || '').replace(/[<>&]/g, '')}</span>`;
    setActions(false);
  } else {
    player.className = 'player empty';
    player.innerHTML = `<span>生成中... ${job.status || ''}</span>`;
    setActions(false);
  }
}

async function poll(jobId) {
  const res = await fetch(`/api/jobs/${jobId}`);
  const job = await res.json();
  renderJob(job);
  history.replaceState(null, '', `?job=${encodeURIComponent(jobId)}`);
  if (job.status === 'done' || job.status === 'error') {
    clearInterval(pollTimer);
    pollTimer = null;
    await loadHistory();
  }
}

$('generate').addEventListener('click', async () => {
  const url = $('source-url').value.trim();
  if (!url) return message('URLを入力してください');
  $('generate').disabled = true;
  message('生成ジョブを開始しています...');
  try {
    const sameLoadedUrl = currentJobId && currentJobUrl && url === currentJobUrl;
    const endpoint = sameLoadedUrl ? `/api/jobs/${encodeURIComponent(currentJobId)}/regenerate` : '/api/jobs';
    const res = await fetch(endpoint, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({url, vtuber_mode:true, video_style:'ai_avatar_explainer'})});
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || '生成開始に失敗しました');
    currentJobId = data.job_id;
    currentJobUrl = url;
    message(`${sameLoadedUrl ? '上書き再生成' : 'ジョブ開始'}: ${currentJobId}`);
    clearInterval(pollTimer);
    await poll(currentJobId);
    pollTimer = setInterval(() => poll(currentJobId), 5000);
  } catch (e) {
    message(e.message || String(e));
  } finally {
    $('generate').disabled = false;
  }
});

function shareText() {
  return `${$('title').textContent}\n${$('summary').textContent}\n${$('kurage-link').href}`;
}

$('copy').addEventListener('click', async () => {
  const text = shareText();
  await navigator.clipboard.writeText(text);
  message('コピーしました');
});

$('post-x').addEventListener('click', () => {
  const text = shareText();
  window.open(`https://x.com/intent/tweet?text=${encodeURIComponent(text)}`, '_blank', 'noopener');
});

$('delete').addEventListener('click', async () => {
  if (!currentJobId || !confirm('この生成ジョブとKurage動画を削除しますか？')) return;
  await fetch(`/api/jobs/${currentJobId}`, {method:'DELETE'});
  currentJobId = null;
  currentJobUrl = '';
  clearInterval(pollTimer);
  $('status').textContent = '削除済み';
  $('title').textContent = 'タイトルはここに表示されます';
  $('summary').textContent = '動画解析後、要点と考察がここに表示されます。';
  $('script').innerHTML = '';
  $('player-wrap').className = 'player empty';
  $('player-wrap').innerHTML = '<span>生成完了後に再生窓が表示されます</span>';
  setActions(false);
  await loadHistory();
});

async function loadHistory() {
  const res = await fetch('/api/jobs');
  const data = await res.json();
  const box = $('history');
  box.innerHTML = '';
  for (const job of data.jobs || []) {
    const div = document.createElement('div');
    div.className = 'history-item';
    const title = escapeHtml(jobTitle(job));
    const url = escapeHtml(job.url || '');
    const status = escapeHtml(`${statusLabel(job)} / ${job.progress ?? 0}%`);
    const kurage = job.kurage_job_id ? `<small>Kurage: ${escapeHtml(job.kurage_status || '-') } / ${escapeHtml(job.kurage_progress ?? '-') }%</small>` : '';
    div.innerHTML = `
      <button class="history-main" data-id="${escapeHtml(job.id)}" type="button">
        <strong>${title}</strong>
        <small>${status} / ${url}</small>
        ${kurage}
      </button>`;
    div.querySelector('button').addEventListener('click', async () => {
      currentJobId = job.id;
      await poll(job.id);
      clearInterval(pollTimer);
      if (!['done', 'error'].includes(job.status)) {
        pollTimer = setInterval(() => poll(job.id), 5000);
      }
    });
    box.appendChild(div);
  }
  if (!box.innerHTML) box.innerHTML = '<div class="message">まだ生成履歴がありません。</div>';
}
$('reload').addEventListener('click', loadHistory);
$('source-url').addEventListener('input', () => {
  if ($('source-url').value.trim() !== currentJobUrl) {
    currentJobId = null;
  }
});

async function openInitialJob() {
  await loadHistory();
  const params = new URLSearchParams(location.search);
  const jobId = params.get('job');
  if (jobId) {
    await poll(jobId);
    clearInterval(pollTimer);
    pollTimer = setInterval(() => poll(jobId), 5000);
  }
}

openInitialJob();
