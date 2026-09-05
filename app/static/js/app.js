// StudyMate AI — shared helpers
// (page-specific logic lives inline in each template)

// Highlight active nav on hash-less routes & small niceties
document.addEventListener('DOMContentLoaded', () => {
  // Dismissible flash messages, if any get added later
  document.querySelectorAll('[data-dismiss]').forEach(btn => {
    btn.addEventListener('click', () => btn.closest('[data-flash]')?.remove());
  });
});
