const app = document.getElementById("app");

if (app) {
  const title = document.createElement("h1");
  title.textContent = "Hello from TypeScript!";

  const description = document.createElement("p");
  description.textContent = "This is a basic TS-powered webpage.";

  const info = document.createElement("pre");
  info.textContent = `Environment: ${navigator.userAgent}`;

  app.appendChild(title);
  app.appendChild(description);
  app.appendChild(info);
}


