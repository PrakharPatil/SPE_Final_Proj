document.addEventListener("DOMContentLoaded", function () {
  // Handle initial page load with hash
  handleHashScroll();

  // Handle browser back/forward navigation
  window.addEventListener("popstate", handleHashScroll);

  // Smooth scroll for same-page navigation
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener("click", function (e) {
      e.preventDefault();
      const hash = this.getAttribute("href");
      history.pushState(null, null, hash);
      scrollToSection(hash);
    });
  });
});

function handleHashScroll() {
  if (window.location.hash) {
    scrollToSection(window.location.hash);
  }
}

// function scrollToSection(hash) {
//   const target = document.querySelector(hash);
//   if (target) {
//     const offset = 100; // Navbar height
//     const position = target.offsetTop - offset;

//     window.scrollTo({
//       top: position,
//       behavior: "smooth",
//     });
//   }
// }
