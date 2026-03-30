const chat = document.getElementById('chat');
const msg = document.getElementById('msg');
const send = document.getElementById('send');

function escapeHtml(s) {
  return (s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function addLine(who, text) {
  const div = document.createElement('div');
  div.className = 'msg ' + who;

  const label = who === 'user' ? 'You' : 'Bot';
  div.innerHTML = `<span class="who">${label}:</span> <span class="text">${escapeHtml(text)}</span>`;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

async function ask() {
  const text = msg.value.trim();
  if (!text) return;
  addLine('user', text);
  msg.value = '';

  try {
    const res = await fetch('http://localhost:8000/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text })
    });
    const data = await res.json();
    addLine('bot', data.answer || JSON.stringify(data));

    if (data.sources && data.sources.length) {
      const src = data.sources.map(s => {
        const u = s.url ? ` (${s.url})` : '';
        return `- ${s.file_name}${u}`;
      }).join('\n');
      addLine('bot', `Sources:\n${src}`);
    }
  } catch (e) {
    addLine('bot', 'Error connecting to backend. Is it running on http://localhost:8000 ?');
  }
}

send.onclick = ask;
msg.addEventListener('keydown', (e) => { if (e.key === 'Enter') ask(); });
