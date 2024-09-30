import json
import time

import requests
from bs4 import BeautifulSoup

# Base URL and pages to scrape
base_url = "https://www.deep-ml.com"
pages_to_scrape = [1, 2, 3]


def fetch_with_retry(url, max_retries=5, timeout=30):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()  # Raise an error for HTTP errors
            return response  # Return the response if successful
        except requests.exceptions.Timeout:
            print(f"Attempt {attempt + 1}/{max_retries} timed out. Retrying...")
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            break  # Break the loop for non-timeout errors
        time.sleep(5)  # Optional: wait before the next retry


# Function to scrape a single page
def scrape_page(page_num, problems_data):
    url = f"{base_url}/?page={page_num}"
    # response = requests.get(url)
    response = fetch_with_retry(url, max_retries=5, timeout=30)
    # Check if the request was successful
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")

        # Find the table body by id
        table_body = soup.find("tbody", id="problemsContainer")

        # If the table body is found, extract all rows
        if table_body:
            rows = table_body.find_all("tr", class_="problem-row")

            # Loop over each row and extract the data
            for row in rows:
                problem_id = row.find("td").text.strip()
                data_difficulty = row.get("data-difficulty")
                data_category = row.get("data-category")

                # Extract the URL from the <a> tag within the second <td>
                link_tag = row.find("a")
                if link_tag:
                    relative_url = link_tag["href"]
                    full_url = base_url + relative_url

                    # Store the extracted information in the dictionary
                    problems_data[problem_id] = {
                        "difficulty": data_difficulty,
                        "category": data_category,
                        "url": full_url,
                        "page": page_num,
                    }
    else:
        print(f"Failed to retrieve page {page_num}")
    return problems_data


# Function to scrape a single problem page
def scrape_problem_page(problem_id, problems_data):
    problem_url = problems_data[problem_id]["url"]
    # response = requests.get(problem_url)
    response = fetch_with_retry(problem_url, max_retries=5, timeout=30)
    print(f"Processing problem: {problem_id}, response: {response.status_code}")
    # Check if the request was successful
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")

        problem_section = soup.find("div", id="problemSection")

        title = problem_section.find("div", class_="card-header").find("h2")
        if title:
            title = title.text.strip()

        description = problem_section.find("div", class_="card-body").find("div")
        if description:
            description = description.text.strip()

        # Extract the example
        example = problem_section.find("pre", class_="bg-light")
        if example:
            example = example.text.strip()
        else:
            example = "N/A"

        # Extract the learn more section
        learn_more = soup.find("div", id="learnSection")
        if learn_more:
            learn_more = learn_more.text.strip()
        else:
            learn_more = "N/A"

        # Find the textarea element by ID
        code_area = soup.find("textarea", {"id": "codeArea"})

        # Extract the Python function code
        python_code = "Code not found"
        if code_area:
            python_code = code_area.text
            # Replace HTML encoded symbols back to their original form
            python_code = python_code.replace("&gt;", ">").replace("&lt;", "<")
        problems_data[problem_id]["title"] = title
        problems_data[problem_id]["description"] = description
        problems_data[problem_id]["example"] = example
        problems_data[problem_id]["learn_more"] = learn_more
        problems_data[problem_id]["python_code"] = python_code
    else:
        print(f"Failed to retrieve page {problem_url}")
        return None
    return problems_data


def save_problem(problem_id, problems_data, out_path="src/unsolved"):
    scrape_problem_page(problem_id, problems_data)
    details = problems_data[problem_id]
    problem = f'''
"""
problem_id: {problem_id}
Category: {details['category']}
Difficulty: {details['difficulty']}
URL: {details['url']}
Page: {details['page']}

==== Title ====
{details['title']}

==== Description ====
{details['description']}

==== Example ====
{details['example']}

==== Learn More ====
{details['learn_more']}
"""

# ==== Code ====
{details["python_code"]}


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1

    # Test case 2

    # Test case 3
'''
    out_file = f"{out_path}/{details['category']}/{problem_id}_{details['title']}.py"
    with open(out_file, "w", encoding="utf-8") as file:
        file.write(problem)


def main():
    # Dictionary to store the extracted data
    problems_data = {}

    # Loop through each page and scrape data
    for page in pages_to_scrape:
        problems_data = scrape_page(page, problems_data)

    # Save problems index dict
    with open("data/problems_index.json", "w") as outfile:
        json.dump(problems_data, outfile)

    out_path = "src/unsolved"
    # Loop the dict and create the correspondent python file
    for problem_id in problems_data.keys():
        time.sleep(2)
        save_problem(problem_id, problems_data, out_path=out_path)


if __name__ == "__main__":
    main()
