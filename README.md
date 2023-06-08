# COVID_webapp
### Development repository for COVID-19 hospitalization webapp

### Summary
This webapp provides a forecasted hospitalization numbers for COVID-19 across a state given user defined parameters. It allows the user to upload a file containing the daily total positive COVID-19 testing numbers across a state, or from a healthcare provider servicing a known % of the state's population. And with the user selecting various epidimiological parameters (e.g., fraction of infected that are symptomatic, months for loss of immunity, state mandated social distancing dates and durations, etc.), the app calls on a Markov Chain Monte Carlo (MCMC) simulation to fit the data to an SIR and forecast the hospitalization rates and uncertainties over the following 100 days.

---

### Design and Functionality

This webapp is written almost entirely in Python (with some HTML and Javascript) using the [Flask](https://flask.palletsprojects.com/en/2.3.x/) to convert incoming HTTP requests to standard WSGI environ, and outgoing WSGI responses to HTTP. It is designed to run on Amazon's AWS using [Boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) to manage the services. The app requires three instances, one to host the webapp, one to perform plotting duties, and one to run the computationally instensive MCMC fitting and predictions. The code also uses [Redis](https://redis.io/docs/clients/python/) for queueing jobs, [Paramiko](https://www.paramiko.org/) for SSH duties, and Python's [email](https://docs.python.org/3/library/email.html#module-email), [smtplib](https://docs.python.org/3/library/smtplib.html), and [ssl](https://docs.python.org/3/library/ssl.html#module-ssl) libraries for sending emails with notifications to the admins and final results to the users. The MCMC ensemble sampler is performed using the Python package [emcee](https://emcee.readthedocs.io/en/stable/).
