# AI Usage Report

> **Statement of Authorship**: In this project, AI tools were utilized exclusively as a technical support system to facilitate development workflows. They did not function as autonomous decision-makers or primary code generators. The entire conceptual framework, including architectural design, model selection, and algorithmic logic, was strictly defined, implemented, and validated by myself to ensure complete mastery of the subject matter.

Here is a list of how the AI tools (Google Gemini, GitHub Copilot, Claude AI) were used in this project:

- **Documentation, Style, & Code Review**:
    - Restructuring and making `README.md` more readable - **Gemini AI**
    - Helping with readability and reproducibility of the code via suggestions - **GitHub Copilot**, **Gemini AI**
    - Improving visualization of the results - **Claude AI**
    - Identifying the problem with the CI/CD pipeline and helping to fix it - **Gemini AI**

- **Debugging & Problem Solving**:
    - Understanding error messages - **GitHub Copilot**, **Gemini AI**
    - Suggestions and help regarding the caching mechanism and resolving Yahoo Finance API issues for altcoins - **Gemini AI**
    - Improving the trading strategy via different thresholds for different assets (comparing results and adapting the strategy) - **Gemini AI**
    - Improving the backtesting process - **Gemini AI**
    - Help with understanding the libraries needed in the project - **Claude AI**
    - Assistance in plugging the Deep Learning model - **Claude AI**

- **Code Architecture & Hygiene**:
    - Help with comments, function names, and docstrings to make code as readable and obvious as possible - **GitHub Copilot** (occasionally autocompleted to save time)
    - Help with repository cleanup and organization - **Gemini AI**

The suggestions were always read carefully and sufficiently in order to understand and implement in the best way possible for the success of this project.

## Technical Challenge Example
A specific technical hurdle was managing **API Rate Limits** from Yahoo Finance. While AI tools proposed generic retry logic, I decided to implement a persistent **local caching system** (saving raw CSVs). This architectural decision significantly accelerated the experimental loop and prevented IP bans, demonstrating how AI advice was filtered through practical engineering judgment.
