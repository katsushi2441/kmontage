let currentJobId = null;
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

function renderJob(job) {
  currentJobId = job.id;
  $('status').textContent = `${job.status || 'unknown'} ${job.progress ?? 0}%`;
  $('title').textContent = job.kurage_title || job.title || '生成中';
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
    player.innerHTML = `<iframe src="https://kurage.exbridge.jp/kuragev.php?id=${job.kurage_job_id}" allow="autoplay; fullscreen" loading="lazy"></iframe>`;
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
    const res = await fetch('/api/jobs', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({url, vtuber_mode:true, video_style:'ai_avatar_explainer'})});
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || '生成開始に失敗しました');
    currentJobId = data.job_id;
    message(`ジョブ開始: ${currentJobId}`);
    clearInterval(pollTimer);
    await poll(currentJobId);
    pollTimer = setInterval(() => poll(currentJobId), 5000);
  } catch (e) {
    message(e.message || String(e));
  } finally {
    $('generate').disabled = false;
  }
});

$('copy').addEventListener('click', async () => {
  const text = `${$('title').textContent}\n${$('summary').textContent}\n${$('kurage-link').href}`;
  await navigator.clipboard.writeText(text);
  message('コピーしました');
});

$('post-x').addEventListener('click', () => {
  const text = `${$('title').textContent}\n${$('kurage-link').href}`;
  window.open(`https://x.com/intent/tweet?text=${encodeURIComponent(text)}`, '_blank', 'noopener');
});

$('delete').addEventListener('click', async () => {
  if (!currentJobId || !confirm('この生成ジョブとKurage動画を削除しますか？')) return;
  await fetch(`/api/jobs/${currentJobId}`, {method:'DELETE'});
  currentJobId = null;
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
    div.innerHTML = `<div><strong>${job.kurage_title || job.title || job.url}</strong><small>${job.status} / ${job.url || ''}</small></div><button data-id="${job.id}">表示</button>`;
    div.querySelector('button').addEventListener('click', async () => {
      currentJobId = job.id;
      await poll(job.id);
    });
    box.appendChild(div);
  }
}
$('reload').addEventListener('click', loadHistory);
loadHistory();
