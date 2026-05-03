const form = document.getElementById("newsForm");
const dateInput = document.getElementById("dateInput");
const error = document.getElementById("error");
const newsContainer = document.getElementById("newsContainer");
const loading = document.getElementById("loading");

form.addEventListener("submit", function(e) {
    e.preventDefault();

    error.innerText = "";
    newsContainer.innerHTML = "";

    // ✅ Form Validation
    if (dateInput.value === "") {
        error.innerText = "Please select a date!";
        return;
    }

    loading.innerText = "Loading news...";

    // ✅ setTimeout
    setTimeout(() => {
        fetchNews()
            .then(showNews)
            .catch(() => {
                error.innerText = "Failed to fetch news.";
            })
            .finally(() => {
                loading.innerText = "";
            });
    }, 1000);
});


// ✅ Fetch API + Promise
function fetchNews() {
    return fetch("https://api.spaceflightnewsapi.net/v4/articles/")
        .then(response => response.json())
        .then(data => data.results);
}


// ✅ DOM Manipulation
function showNews(articles) {

    articles.slice(0, 5).forEach(article => {

        const div = document.createElement("div");
        div.classList.add("news");

        div.innerHTML = `
            <h3>${article.title}</h3>
            <p>${article.summary.substring(0,100)}...</p>
            <button onclick="openGoogle('${article.title}')">
                Read More on Google
            </button>
        `;

        newsContainer.appendChild(div);
    });
}


// ✅ Google Navigation
function openGoogle(title) {
    const query = encodeURIComponent(title);
    window.open(`https://www.google.com/search?q=${query}`, "_blank");
}