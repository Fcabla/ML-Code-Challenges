import json


from problem_scrapper import save_problem, scrape_page

# Base URL and pages to scrape
base_url = "https://www.deep-ml.com"
pages_to_scrape = [1, 2, 3]

# Dictionary to store the extracted data
problems_data = {}

with open("data/problems_index.json", "r") as outfile:
    problem_index = json.load(outfile)

# Loop through each page and scrape data
for page in pages_to_scrape:
    scrape_page(page, problems_data)

missing_keys = set(problems_data.keys()) - set(problem_index.keys())

if missing_keys:
    for key in missing_keys:
        problem_index[key] = problems_data[key]
        save_problem(key, problems_data)
    # Save problems index dict
    with open("data/problems_index.json", "w") as outfile:
        json.dump(problems_data, outfile)
