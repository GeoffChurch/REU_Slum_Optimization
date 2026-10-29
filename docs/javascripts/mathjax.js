// MathJax v3's configuration for pymdownx.arithmatex (`generic: true`, mkdocs.yml): typeset only
// what arithmatex marked, with its \( \) and \[ \] delimiters, and again whenever Material swaps a
// page in. Loaded before the MathJax bundle itself (mkdocs.yml extra_javascript order). It lives
// here, not in docs/js/, because docs/js/ is the web build's output and is gitignored: the first
// copy was written there and never reached the repository, so every page requested a 404.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
