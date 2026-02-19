module.exports = {
  content: ["_site/**/*.html", "_site/**/*.js"],
  css: ["_site/assets/css/*.css"],
  output: "_site/assets/css/",
  skippedContentGlobs: ["_site/assets/**/*.html"],
  // Keep social icon classes that jekyll-socials injects (plugin output may not be seen by PurgeCSS in some builds)
  safelist: ["fa-brands", "fa-linkedin", "fa-github", "fa-google-scholar", "ai-google-scholar"],
};
