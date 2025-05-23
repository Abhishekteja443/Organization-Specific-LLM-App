let baseUrls = [];
let extraUrls = [];


function isValidURL(url) {
    // Regular expression to check if the input is a valid URL
    let urlPattern = /^(https?:\/\/)?([a-zA-Z0-9.-]+)\.[a-zA-Z]{2,6}(\/\S*)?$/;
    return urlPattern.test(url);
}

function addBaseURL() {
    let baseUrlInput = document.getElementById("base-url");
    let url = baseUrlInput.value.trim();

    if (!url) {
        showError("Please enter a valid Base URL.");
        return;
    }
    if (!isValidURL(url)) {
        showError("Base URL must be a valid URL format (http/https).");
        return;
    }

    if (!url.toLowerCase().includes("sitemap")) {
        showError("Base URL must contain the word 'sitemap'.");
        return;
    }

    if (baseUrls.includes(url)) {
        showError("This Base URL is already added.");
        return;
    }

    baseUrls.push(url);
    updateBaseUrlList();
    baseUrlInput.value = "";
}

function addExtraURL() {
    let extraUrlInput = document.getElementById("extra-url");
    let url = extraUrlInput.value.trim();

    if (url.toLowerCase().includes("sitemap")) {
        showError("URL must contain non sitemap urls.");
        return;
    }

    if (!url) {
        showError("Please enter a valid Additional URL.");
        return;
    }
    if (!isValidURL(url)) {
        showError("Additional URL must be a valid URL format (http/https).");
        return;
    }

    if (extraUrls.includes(url)) {
        showError("This Additional URL is already added.");
        return;
    }

    extraUrls.push(url);
    updateExtraUrlList();
    extraUrlInput.value = "";
}

function updateBaseUrlList() {
    let list = document.getElementById("base-url-list");
    list.innerHTML = "";
    baseUrls.forEach((url, index) => {
        list.innerHTML += `<li>${url} <button onclick="removeBaseUrl(${index})">X</button></li>`;
    });
}

function updateExtraUrlList() {
    let list = document.getElementById("extra-url-list");
    list.innerHTML = "";
    extraUrls.forEach((url, index) => {
        list.innerHTML += `<li>${url} <button onclick="removeExtraUrl(${index})">X</button></li>`;
    });
}

function removeBaseUrl(index) {
    baseUrls.splice(index, 1);
    updateBaseUrlList();
}

function removeExtraUrl(index) {
    extraUrls.splice(index, 1);
    updateExtraUrlList();
}

function showError(message) {
    let errorMessage = document.getElementById("error-message");
    errorMessage.innerText = message;
    setTimeout(() => {
        errorMessage.innerText = "";
    }, 3000);
}

function displayOutput(message, submittedUrls, unscrapedUrls) {
    let outputDiv = document.getElementById("output");
    
    let html = `<strong>${message}</strong><br><br>`;
    
    // Display submitted URLs
    html += `<h3>Submitted URLs (${submittedUrls.length}):</h3>`;
    html += `<ul>${submittedUrls.map(url => `<li>${url}</li>`).join('')}</ul>`;
    
    // Display unscraped URLs
    html += `<h3>Unscraped URLs (${unscrapedUrls.length}):</h3>`;
    html += `<ul>${unscrapedUrls.map(url => `<li>${url}</li>`).join('')}</ul>`;
    
    // Add Organization-GPT button
    html += `<button id="org-gpt-btn" class="submit-btn" onclick="window.location.href='/organization-gpt'">Organization-GPT</button>`;
    
    outputDiv.innerHTML = html;
    outputDiv.style.display = "block";
}

function submitData() {
    if (baseUrls.length === 0 && extraUrls.length === 0) {
        showError("At least one URL (Base or Extra) must be provided.");
        return;
    }

    fetch('/submit-urls', {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ base_urls: baseUrls, extra_urls: extraUrls })
    })
    .then(response => response.json())
    .then(data => {
        // Pass all three pieces of information to displayOutput
        displayOutput(data.message, data.submitted_urls, data.unscraped_urls);
        baseUrls = [];
        extraUrls = [];
        updateBaseUrlList();
        updateExtraUrlList();
    })
    .catch(error => {
        showError("Error submitting URLs");
        console.error(error);
    });
}
