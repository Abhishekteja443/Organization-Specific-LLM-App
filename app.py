from flask import Flask, render_template, request, jsonify
from src import helper

app = Flask(__name__)

  # Store submitted URLs to prevent duplicates

@app.route("/")
def admin_panel():
    return render_template("index.html")

@app.route("/submit-urls", methods=["POST"])
def receive_urls():
    stored_urls = set()
    data = request.json
    base_urls = data.get("base_urls", [])
    extra_urls = set(data.get("extra_urls", []))
    stored_urls= stored_urls| extra_urls


    for i in base_urls:
        stored_urls=stored_urls | helper.fetch_urls_from_sitemap(i)
    
    

    unscraped_urls=helper.process_urls(stored_urls)

    print(f"Received Base URLs: {base_urls}")
    print(f"Received Extra URLs: {extra_urls}")

    result_message = f"Received {len(base_urls)} Base URLs and {len(extra_urls)} Additional URLs."
    
    return jsonify({"message": result_message, "submitted_urls": list(stored_urls)})


if __name__ == "__main__":
    app.run(debug=True)
