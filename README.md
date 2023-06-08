# COVID_webapp
### Development repository for COVID-19 hospitalization webapp

### Summary
This webapp provides a forecasted hospitalization numbers for COVID-19 across a state given user defined parameters. It allows the user to upload a file containing the daily total positive COVID-19 testing numbers across a state, or from a healthcare provider servicing a known % of the state's population. And with the user selecting various epidimiological parameters (e.g., fraction of infected that are symptomatic, months for loss of immunity, state mandated social distancing dates and durations, etc.), the app calls on a Markov Chain Monte Carlo (MCMC) simulation to fit the data to an SIR and forecast the hospitalization rates and uncertainties over the following 100 days.

---

