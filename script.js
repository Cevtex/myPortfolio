// Shared site script: theme + PDF (offline-first, Waterfox-friendly)
(function () {
  const html = document.documentElement;
  const toggle = document.getElementById('themeToggle');
  const icon = document.getElementById('themeIcon');

  const stored = localStorage.getItem('theme') || 'dark';
  html.setAttribute('data-theme', stored);
  if (icon) icon.textContent = stored === 'dark' ? '🌙' : '☀️';

  if (toggle) {
    toggle.addEventListener('click', () => {
      const current = html.getAttribute('data-theme') || 'dark';
      const next = current === 'dark' ? 'light' : 'dark';
      html.setAttribute('data-theme', next);
      localStorage.setItem('theme', next);
      if (icon) icon.textContent = next === 'dark' ? '🌙' : '☀️';
    });
  }

  function filenameForPage() {
    const fromAttr = document.body.getAttribute('data-pdf-filename');
    if (fromAttr) return fromAttr;
    // sanitize title
    const t = (document.title || 'Portfolio').replace(/[\\/:*?"<>|]/g, '');
    return t + '.pdf';
  }

  function pdfTargetElement() {
    const sel = document.body.getAttribute('data-pdf-target');
    if (sel) {
      const el = document.querySelector(sel);
      if (el) return el;
    }
    return document.body;
  }

  function printIframeIfPresent() {
    const frame = document.getElementById('nbFrame');
    if (frame && frame.contentWindow) {
      frame.contentWindow.focus();
      frame.contentWindow.print();
      return true;
    }
    return false;
  }

  const pdfBtn = document.getElementById('pdfDownload');
  if (pdfBtn) {
    pdfBtn.addEventListener('click', () => {
      // Special case: notebook viewer prints the iframe.
      if (document.body.getAttribute('data-pdf-mode') === 'iframe') {
        if (printIframeIfPresent()) return;
      }

      if (typeof html2pdf === 'undefined') {
        // Fallback always works (Waterfox safe)
        window.print();
        return;
      }

      // Hide nav for cleaner PDF
      document.body.classList.add('pdf-mode');

      const options = {
        margin: [0.5, 0.5, 0.5, 0.5],
        filename: filenameForPage(),
        image: { type: 'jpeg', quality: 0.98 },
        html2canvas: {
          scale: 2,
          useCORS: true,
          letterRendering: true,
          logging: false
        },
        jsPDF: {
          unit: 'in',
          format: 'letter',
          orientation: 'portrait',
          compress: true
        },
        pagebreak: {
          mode: ['avoid-all', 'css', 'legacy']
        }
      };

      try {
        html2pdf().from(pdfTargetElement()).set(options).save().then(() => {
          document.body.classList.remove('pdf-mode');
        }).catch(() => {
          document.body.classList.remove('pdf-mode');
          window.print();
        });
      } catch (e) {
        document.body.classList.remove('pdf-mode');
        window.print();
      }
    });
  }
})();