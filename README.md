# COVID_webapp
### Development repository for COVID-19 hospitalization webapp

### Summary
This webapp provides a forecasted hospitalization numbers for COVID-19 across a state given user defined parameters. It allows the user to upload a file containing the daily total positive COVID-19 testing numbers across a state, or from a healthcare provider servicing a known % of the state's population. And with the user selecting various epidimiological parameters (e.g., fraction of infected that are symptomatic, months for loss of immunity, state mandated social distancing dates and durations, etc.), the app calls on a Markov Chain Monte Carlo (MCMC) simulation to fit the data to an SIR and forecast the hospitalization rates and uncertainties over the following 100 days.

---

### Design and Functionality

This webapp is written almost entirely in Python (with some HTML and Javascript) using [Flask](https://flask.palletsprojects.com/en/2.3.x/) to convert incoming HTTP requests to standard WSGI environ, and outgoing WSGI responses to HTTP. It is designed to run on Amazon's AWS using [Boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) to manage the services. The app requires three instances, one to host the webapp, one to perform plotting duties, and one to run the computationally instensive MCMC fitting and predictions. The code also uses [Redis](https://redis.io/docs/clients/python/) for queueing jobs, [Paramiko](https://www.paramiko.org/) for SSH duties, and Python's [email](https://docs.python.org/3/library/email.html#module-email), [smtplib](https://docs.python.org/3/library/smtplib.html), and [ssl](https://docs.python.org/3/library/ssl.html#module-ssl) libraries for sending emails with notifications to the admins and final results to the users. The MCMC ensemble sampler is performed using the Python package [emcee](https://emcee.readthedocs.io/en/stable/).

The webapp works by allowing the user to upload a .csv with either a state's daily hospitalization numbers, or a healthcare provider (with pre-known % of population served) detailed daily hospitalizations. It also allows the user to select a number of epidemiological parameters to control the model. The user can then run the full simulation using Markov Chain Monte Carlo (MCMC) to produce the model, predicted values for the near future, and ranges of uncertainty. They also have the option to obtain a quick plot of the model and the uploaded data based on their selected parameters (without future predictions) to confirm that the model matches the historic data prior to running the lengthy simulation. Since the full simulation can take several minutes and up to hours depending on the user selected intensity, the user is asked to provide an email address to which the webapp will send the simulation results once completed. All data is purged from the system once the job is completed, and the final results are retained for only 3 days before purging.


