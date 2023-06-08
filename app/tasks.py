from app import app
import time
from time import strftime
import os
import numpy as np
import pandas as pd
import emcee
import corner
from scipy.optimize import minimize
import random
import csv
import matplotlib.pyplot as plt
import re
import subprocess

import boto3
import paramiko

import email, smtplib, ssl
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage


def test_email(email):
    sender_email = app.config['MAIL_USERNAME']
    password = app.config['MAIL_PASSWORD']
    receiver_email = email
    public_ipv4 = str(subprocess.check_output('ec2metadata --public-ipv4', shell=True)).split("'")[1].split('\\n')[0]

    subject = "COVID task completed"

    body = f"""Hello,

        This email is to confirm that the service is working.

        Thank you,
        The COVID team
        """
    # Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject

    # Add body to email
    message.attach(MIMEText(body, "plain"))

    text = message.as_string()

    context = ssl.create_default_context()

    # Try to log in to server and send email
    try:
        #with smtplib.SMTP_SSL(app.config['MAIL_SERVER'], app.config['MAIL_PORT'], context=context) as server:
        #    server.login(app.config['MAIL_USERNAME'], app.config['MAIL_PASSWORD'])
        #    server.sendmail(
        #        app.config['MAIL_USERNAME'], receiver_email, text
        #    )
        with smtplib.SMTP_SSL(app.config['MAIL_SERVER'], app.config['MAIL_PORT'], context=context) as server:
            server.login(sender_email, password)
            server.sendmail(
                sender_email, receiver_email, text
            )
        print(f'email successfully sent from {sender_email} to {receiver_email}')
    except Exception as e:
        # Print any error messages to stdout
        print('******')
        print('ERROR in sending email')
        print(e)
        print('******')

    return email


def test_script():
   # testing start and stop remote instance
    ec2Resource = boto3.resource('ec2')
    ec2Client = boto3.client('ec2')

    compInstId = 'i-0d7ef00fc77e3e14f'
    remote_instance = ec2Resource.Instance(compInstId)
    if remote_instance.state['Name'] == 'running':
        print('Instance is running')
    else:
        ec2Client.start_instances(InstanceIds=[compInstId])
        time.sleep(20)#

    print(remote_instance.state['Name'])

    reservations = ec2Client.describe_instances(InstanceIds=[compInstId]).get("Reservations")
    for reservation in reservations:
        for instance in reservation['Instances']:
            public_ipv4 = instance.get("PublicIpAddress")

    public_ipv4 = ec2Client.describe_instances(InstanceIds=compInstId).get("PublicIpAddress")
    print("IP = ", public_ipv4)

    ec2Client.stop_instances(InstanceIds=[compInstId])

    return compInstId


def create_configs_file(DEorCC, npop, nsteps, fracwith, daysuntils, infectious_days,
                        symptoms_days_before_hospital, reinsert, vaccinated,
                        DistancingDates, DistancingDurations, DistancingExtent, filecode):
    '''takes the user input from webapp form,
    creates webapp_configs.csv file'''

    if DistancingDates == []:
        distancingdates = 'NA'
    else:
        distancingdates = ','.join(map(str, DistancingDates))

    if DistancingDurations == []:
        distancingdurations = 'NA'
    else:
        distancingdurations = ','.join(map(str, DistancingDurations))

    if DistancingExtent == []:
        distancingextent = 'NA'
    else:
        distancingextent = ','.join(map(str, DistancingExtent))

    header = '\n'.join(
        [line for line in
            ['DEorCC,' + str(DEorCC),
             'npop,' + str(npop),
             'nsteps,' + str(nsteps),
             'fracwith,' + str(fracwith),
             'daysuntils,' + str(daysuntils),
             'infectious_days,' + str(infectious_days),
             'symptoms_days_before_hospital,' + str(symptoms_days_before_hospital),
             'reinsert,' + str(reinsert),
             'vaccinated,' + str(vaccinated),
             'DistancingDates,' + distancingdates,
             'DistancingDurations,' + distancingdurations,
             'DistancingExtent,' + distancingextent
            ]
        ]
    )

    filename = "webapp_configs_" + str(filecode) + ".csv"
    filepath = os.path.join(app.config["CSV_CONFIGS"], filename)
    with open(filepath, 'w', newline='') as csvfile:
        for line in header:
            csvfile.write(line)

    return filepath


def quick_plot_remote(DEorCC, npop, nsteps, fracwith, daysuntils, infectious_days,
                      symptoms_days_before_hospital, reinsert, vaccinated,
                      DistancingDates, DistancingDurations, DistancingExtent, 
                      filename, filecode):

    # create webapp_configs.csv file from user input
    webapp_configs = create_configs_file(DEorCC, npop, nsteps, fracwith, daysuntils, infectious_days,
                                         symptoms_days_before_hospital, reinsert, vaccinated,
                                         DistancingDates, DistancingDurations, DistancingExtent, 
                                         filecode)

    ec2Instances = boto3.resource('ec2')

    remote_instance = ec2Instances.Instance('i-') #insert COVID Plotting t2.micro instance id
    #remote_instance = ec2Instances.Instance('i-') #insert development instance id

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(remote_instance.public_dns_name, username='ubuntu', key_filename='../../covid_dev/covid_dev_key_pair.pem')

    # open sftp channel to compute instance
    sftp = ssh.open_sftp()

    # send positives.csv file to compute instance
    filepath = os.path.join(app.config["CSV_UPLOADS"], filename)
    sftp.put(filepath, './positives.csv')
    print('sftp of positives.csv file successful')

    # send webapp_configs.csb file to compute instance
    sftp.put(webapp_configs, './webapp_configs.csv')
    print('sftp of webapp_configs.csv file successful')

    ssh_transp = ssh.get_transport()
    chan = ssh_transp.open_session()
    chan.setblocking(0)

    print('Running covid.py ...')
    command_mcmc = "python3 ./covid/py/covid.py " + DEorCC + " webapp"
    chan.exec_command(command_mcmc)

    # wait for command to complete before moving on to next command
    while True:
        if chan.exit_status_ready():
            break
        time.sleep(1)
    retcode = chan.recv_exit_status()
    print('covid complete')

    # sftp results to img/output directory with filecode
    outdir = app.config["QUICK_PLOT_DIR"]
    remoteplot = str("covid_initialGuess_" + DEorCC + ".png")
    if DEorCC == "DE":
      remoteplot = str("covid_initialGuess_state-wide.png")
    currentplot = str("covid_initialGuess_" + filecode + ".png")
    sftp.get('./'+remoteplot, os.path.join(outdir, currentplot))
    print('Plot copied successfully')

    # cleanup the compute instance
    command_cleanup = "./clean_up.sh"
    stdin, stdout, stderr = ssh.exec_command(command_cleanup)
    print('Remote compute clean_up successful')

    ssh.close()

    return currentplot


def send_file_to_remote(DEorCC, npop, nsteps, fracwith, daysuntils, infectious_days,
                        symptoms_days_before_hospital, reinsert, vaccinated,
                        DistancingDates, DistancingDurations, DistancingExtent,
                        filename, filecode, email, username):
    start = time.time()

    # create webapp_configs.csv file from user input
    webapp_configs = create_configs_file(DEorCC, npop, nsteps, fracwith, daysuntils, infectious_days,
                                         symptoms_days_before_hospital, reinsert, vaccinated,
                                         DistancingDates, DistancingDurations, DistancingExtent, filecode)


    ec2Instances = boto3.resource('ec2')

#    myinst = os.popen('ec2metadata --instance-id')
#    for instance in ec2Instances.instances.all():
#        if instance.id not in myinst:
#            remote_instance = instance

#    remote_instance = ec2Instances.Instance('i-') #insert COVID Dev Compute t2.micro instance id
    remote_instance = ec2Instances.Instance('i-') #insert COVID paid compute instance t2.2xlarge id
#    if remote_instance.state['Name'] == 'running':


    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(remote_instance.public_dns_name, username='ubuntu', key_filename='../../covid_dev/covid_dev_key_pair.pem')

    # open sftp channel to compute instance
    sftp = ssh.open_sftp()

    # send positives.csv file to compute instance
    filepath = os.path.join(app.config["CSV_UPLOADS"], filename)
#    sftp.put(filepath, './input/'+filename)
    sftp.put(filepath, './positives.csv')
    print('sftp of positives.csv file successful')

    # send webapp_configs.csb file to compute instance
    sftp.put(webapp_configs, './webapp_configs.csv')
    print('sftp of webapp_configs.csv file successful')
    time.sleep(3)

    ssh_transp = ssh.get_transport()
    chan = ssh_transp.open_session()
    chan.setblocking(0)

    print('Running covid_mcmc ...')
    command_mcmc = "python3 ./covid/py/covid_mcmc.py " + DEorCC + " webapp"
#    stdin, stdout, stderr = ssh.exec_command(command_mcmc)
    chan.exec_command(command_mcmc)
    # wait for command to complete before moving on to next command
    while True:
        if chan.exit_status_ready():
            break
        time.sleep(5)
    retcode = chan.recv_exit_status()
    print('covid_mcmc complete')

    ssh_transp = ssh.get_transport()
    chan = ssh_transp.open_session()
    chan.setblocking(0)

    print('Running covid_prediction ...')
    command_predict = "python3 ./covid/py/covid_prediction.py " + DEorCC + " webapp"
#    stdin, stdout, stderr = ssh.exec_command(command_predict)
    chan.exec_command(command_predict)
    # wait for command to complete before moving on to next command
    while True:
        if chan.exit_status_ready():
            break
        time.sleep(5)
    retcode = chan.recv_exit_status()
    print('covid_prediction complete')
#    ssh_transp.close()

    ssh_transp = ssh.get_transport()
    chan = ssh_transp.open_session()
    chan.setblocking(0)

    #if DEorCC == "DE":
    #    DEorCC = "state-wide"
    print('Running covid_print_to_file ...')
    command_tofile = "python3 ./covid_print_to_file.py " + DEorCC #+ " " + filecode
#    stdin, stdout, stderr = ssh.exec_command(command_tofile, get_pty=True)
    chan.exec_command(command_tofile)
#    for line in iter(stdout.readline, ""):
#        print(line, end="")

    # wait for command to complete before moving on to next command
    while True:
        if chan.exit_status_ready():
            break
        time.sleep(5)
    retcode = chan.recv_exit_status()
    print('covid_print_to_file complete')
    # create results directory with filecode as name
    resdir = os.path.join(app.config["RESULTS_DIRECTORY"], filecode)
    os.mkdir(resdir)
    print(resdir, " Directory created")

    # names of files needed for results
    if DEorCC == "DE":
        DEorCC = "state-wide"
    predplot = str("predictions_" + DEorCC + ".png")
    predcsv = str("predictions_" + DEorCC + ".csv")
    paramcsv = str("results_" + DEorCC + ".csv")

    # sftp results to filecode results dir
    sftp.get('./'+predplot, os.path.join(resdir, "predictions.png"))
    sftp.get('./'+predcsv, os.path.join(resdir, "predictions.csv"))
    sftp.get('./'+paramcsv, os.path.join(resdir, "parameters.csv"))
    print('Result files copied successfully')

    # cleanup the compute instance
    command_cleanup = "./clean_up.sh"
    stdin, stdout, stderr = ssh.exec_command(command_cleanup)
    print('Remote compute clean_up successful')

#    outfilename = stdout.readlines()[0].split('\n')[0]
#    print(outfilename)
#    sftp.get('./output/'+outfilename, os.path.join(app.config["RESULTS_DIRECTORY"], outfilename))
    ssh.close()

    sender_email = app.config['MAIL_USERNAME']
    password = app.config['MAIL_PASSWORD']
    receiver_email = email
    cc_email = "" # insert email address of admin 1 to cc
    cc_email = "" # insert email address of admin 2 to cc
    ToAddress = [receiver_email, cc_email]
    public_ipv4 = str(subprocess.check_output('ec2metadata --public-ipv4', shell=True)).split("'")[1].split('\\n')[0]
    #r = re.compile('results_(.*?).csv')
    #m = r.search(outfilename)
    #if m:
    #    filecode = m.group(1)

    result_url = 'http://' + str(public_ipv4) + ':5000/results/' + filecode
#    output_url = 'http://' + str(public_ipv4) + ':5000/output-result'

    subject = "COVID task completed"
    if username == "":
        body = f"""Hello,

        Your COVID request has finished its simulation.

        For the next 3 days your results can be viewed at the URL below:
        {result_url}

        Regards,
        The COVID team
        """
    else:
        body = f"""Hello {username},

        Your COVID request has finished its simulation.

        For the next 3 days your results can be viewed at the URL below:
        {result_url}

        Regards,
        The COVID team
        """

    # Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Cc"] = cc_email
#    message["To"] = ', '.join(receiver_email)
#    message["Cc"] = ', '.join(cc_email)

    message["Subject"] = subject

    # Add body to email
    message.attach(MIMEText(body, "plain"))

    # Add plot attachment
    attplt = open(os.path.join(resdir, "predictions.png"), 'rb').read()
    message.attach(MIMEImage(attplt, name="predictions.png"))

    # Open CSV file in binary mode
#    with open(os.path.join(app.config["RESULTS_DIRECTORY"], outfilename), "rb") as attachment:
#        # Add file as application/octet-stream
#        # Email client can usually download this automatically as attachment
#        part = MIMEBase("application", "octet-stream")
#        part.set_payload(attachment.read())

#    # Encode file in ASCII characters to send by email    
#    encoders.encode_base64(part)


    # Add header as key/value pair to attachment part
#    part.add_header(
#        "Content-Disposition",
#        f"attachment; filename= {outfilename}",
#    )
    
    # Add attachment to message and convert message to string
#    message.attach(part)
    text = message.as_string()

    # Create a secure SSL context
    context = ssl.create_default_context()
    
    # Try to log in to server and send email
    try:
        #with smtplib.SMTP_SSL(app.config['MAIL_SERVER'], app.config['MAIL_PORT'], context=context) as server:
        #    server.login(app.config['MAIL_USERNAME'], app.config['MAIL_PASSWORD'])
        #    server.sendmail(
        #        app.config['MAIL_USERNAME'], receiver_email, text
        #    )
        with smtplib.SMTP_SSL(app.config['MAIL_SERVER'], app.config['MAIL_PORT'], context=context) as server:
            server.login(sender_email, password)
            server.sendmail(
#                sender_email, receiver_email, text
                sender_email, ToAddress, text
            )
        print(f'email successfully sent from {sender_email} to {receiver_email}')
    except Exception as e:
        # Print any error messages to stdout
        print('******')
        print('ERROR in sending email')
        print(e)
        print('******')

    end = time.time()
    time_elapsed = end - start
    
    print(f"Job successfully completed")
    print(f"Time elapsed: {time_elapsed} ms")

    return time_elapsed


def extract_results(filepath, filecode):
    df = pd.read_csv(filepath, skiprows=12)
    rowid = 0
    with open(filepath, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if rowid == 2:
                ls_soln = np.array(row).astype(np.float)
            elif rowid == 5:
                ml_soln = np.array(row).astype(np.float)
            elif rowid == 6:
                tmpvar = np.array(row)
                nsteps = tmpvar[-1].astype(np.float)
            elif rowid == 8:
                tmpvar = np.array(row)
                mcmc_m = tmpvar[1:].astype(np.float)
            elif rowid == 9:
                tmpvar = np.array(row)
                mcmc_b = tmpvar[1:].astype(np.float)
            elif rowid == 10:
                tmpvar = np.array(row)
                mcmc_f = tmpvar[1:].astype(np.float)
            rowid += 1
    
    print("Plotting results...")
    plotname = str("best_fit_" + filecode + ".png")
    x0 = np.linspace(0, 10, 500)
    plt.close('all')
    plt.fill_between(x0, np.dot(np.vander(x0, 2), [mcmc_m[5], mcmc_b[5]]), np.dot(np.vander(x0, 2), [mcmc_m[6], mcmc_b[6]]), color='red', alpha=0.2, label="3sigma")
    plt.fill_between(x0, np.dot(np.vander(x0, 2), [mcmc_m[3], mcmc_b[3]]), np.dot(np.vander(x0, 2), [mcmc_m[4], mcmc_b[4]]), color='lightgreen', label="2sigma")
    plt.fill_between(x0, np.dot(np.vander(x0, 2), [mcmc_m[1], mcmc_b[1]]), np.dot(np.vander(x0, 2), [mcmc_m[2], mcmc_b[2]]), color='lightblue', label="1sigma")
    plt.errorbar(df['x'], df['y'], yerr=df['yerr'], fmt=".k", capsize=0)
    plt.plot(x0, np.dot(np.vander(x0, 2), [ls_soln[0], ls_soln[2]]), "--k", label="LS")
    plt.plot(x0, np.dot(np.vander(x0, 2), ml_soln[:-1]), ":k", label="ML")
    plt.plot(x0, np.dot(np.vander(x0, 2), [mcmc_m[0], mcmc_b[0]]), "-b", label="MCMC")
    plt.legend(fontsize=10)
    plt.xlim(0, 10)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(os.path.join(app.config["PLOT_RESULTS"], plotname), bbox_inches='tight')
    plt.close('all')

    return plotname, ls_soln, ml_soln, nsteps, mcmc_m, mcmc_b, mcmc_f, df



def file_cleanup(path):
    now = time.time()

    i = 0
    for f in os.listdir(path):
        ff = os.path.join(path, f)
        if os.stat(ff).st_mtime < now - 3 * 86400:
            os.remove(ff)
            i += 1

    return i


def LS_MS_mcmc_fits(username, email, nsteps, df):
    start = time.time()
    
    x = np.array(df['x'])
    y = np.array(df['y'])
    yerr = np.array(df['yerr'])
    
    # linear least squares solution
    A = np.vander(x, 2)
    C = np.diag(yerr * yerr)
    ATA = np.dot(A.T, A / (yerr ** 2)[:, None])
    cov = np.linalg.inv(ATA)
    w = np.linalg.solve(ATA, np.dot(A.T, y / yerr ** 2))
    
    # Maximum likelihood solution
    nll = lambda *args: -log_likelihood(*args)
    initial = np.array([w[0], w[1], np.log(0.534)]) + 0.1 * np.random.randn(3)
    soln = minimize(nll, initial, args=(x, y, yerr))
    m_ml, b_ml, log_f_ml = soln.x
    
    # MCMC
    pos = soln.x + 1e-4 * np.random.randn(32, 3)
    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr))
    sampler.run_mcmc(pos, nsteps, progress=True)
    
    #samples = sampler.get_chain()
    try:
        tau = sampler.get_autocorr_time()
    except:
        tau = [29.5, 29.1, 28.6]
    
    burnin = int(2 * np.max(tau))
    thin = int(0.5 * np.min(tau))
    
    flat_samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
    
    mcmc_m = np.percentile(flat_samples[:, 0], [50, 16, 84, 2.5, 97.5, 0.15, 99.85])
    mcmc_b = np.percentile(flat_samples[:, 1], [50, 16, 84, 2.5, 97.5, 0.15, 99.85])
    mcmc_f = np.percentile(flat_samples[:, 2], [50, 16, 84, 2.5, 97.5, 0.15, 99.85])
    
    timestamp = str(strftime("%Y%m%d_%H%M%S"))
    randstamp = str(random.randint(0, 100000))
    filename = str("results_" + timestamp + "_" + randstamp + ".csv")
    filepath = os.path.join(app.config["RESULTS_DIRECTORY"], filename)
    
    lsarr = [w[0], np.sqrt(cov[0, 0]), w[1], np.sqrt(cov[1, 1])]
    header = '\n'.join(
        [line for line in 
            ['Least Squares',
             'm,m_err,b,b_err',
             ','.join(map(str, lsarr)),
             'Maximum Likelihood',
             'm,b,log(f)',
             ','.join(map(str, soln.x)),
             'MCMC,' + str(nsteps),
             'var,median,1sigma_m,1sigma_p,2sigma_m,2sigma_p,3sigma_m,3sigma_p',
             'm,' + ','.join(map(str, mcmc_m)),
             'b,' + ','.join(map(str, mcmc_b)),
             'f,' + ','.join(map(str, mcmc_f)),
             'Raw Data',''
            ]
        ]
    )
    
    with open(filepath, 'w', newline='') as csvfile:
        for line in header:
            csvfile.write(line)
        df.to_csv(csvfile, index=False)
    
    
    # -- Send an email with csv file attached
    sender_email = app.config['MAIL_USERNAME']
#    sender_email = os.environ.get('MAIL_USERNAME')
#    password = os.environ.get('MAIL_PASSWORD')
    receiver_email = email
    password = app.config['MAIL_PASSWORD']
    
    subject = "COVID task completed"
    if username == "":
        body = f"""Hello,
    
        Your COVID request has finished its simulation.
        To view the results download the attached .csv, go to our Output page and upload the file there.
        """
    else:
        body = f"""Hello {username},
    
        Your COVID request has finished its simulation.
        To view the results download the attached .csv, go to our Output page and upload the file there.
        """
    
    # Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    
    # Add body to email
    message.attach(MIMEText(body, "plain"))

    # Open CSV file in binary mode
    with open(filepath, "rb") as attachment:
        # Add file as application/octet-stream
        # Email client can usually download this automatically as attachment
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())

    # Encode file in ASCII characters to send by email    
    encoders.encode_base64(part)
    
    # Add header as key/value pair to attachment part
    part.add_header(
        "Content-Disposition",
        f"attachment; filename= {filename}",
    )
    
    # Add attachment to message and convert message to string
    message.attach(part)
    text = message.as_string()

    # Create a secure SSL context
    context = ssl.create_default_context()
    
    # Try to log in to server and send email
    try:
        #with smtplib.SMTP_SSL(app.config['MAIL_SERVER'], app.config['MAIL_PORT'], context=context) as server:
        #    server.login(app.config['MAIL_USERNAME'], app.config['MAIL_PASSWORD'])
        #    server.sendmail(
        #        app.config['MAIL_USERNAME'], receiver_email, text
        #    )
        with smtplib.SMTP_SSL(app.config['MAIL_SERVER'], app.config['MAIL_PORT'], context=context) as server:
            server.login(sender_email, password)
            server.sendmail(
                sender_email, receiver_email, text
            )
        print(f'email successfullt sent from {sender_email} to {receiver_email}')
    except Exception as e:
        # Print any error messages to stdout
        print('******')
        print('ERROR in sending email')
        print(e)
        print('******')
    
    #mail = Mail(app)
    #msg = Message('Hello from the other side!', sender = '', recipients = [email])
    #if username == "":
    #    msg.body = "Hello, your requested job from COVID has successfully completed. " + \
    #                "To view the results download the attached .csv, go to our Output page and upload the file there"
    #else:
    #    msg.body = f"Hello {username}, your requested job from COVID has successfully completed. " + \
    #                "To view the results download the attached .csv, go to our Output page and upload the file there"
    #with app.open_resource(filepath) as fp:
    #    msg.attach(filename, "text/csv", fp.read())  
    #mail.send(msg)
    
    end = time.time()
    time_elapsed = end - start
    
    print(f"Job successfully completed")
    print(f"Time elapsed: {time_elapsed} ms")
    
    return filepath, time_elapsed
    

def log_likelihood(theta, x, y, yerr):
    m, b, log_f = theta
    model = m * x + b
    sigma2 = yerr ** 2 + model ** 2 * np.exp(2 * log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

def log_prior(theta):
    m, b, log_f = theta
    if -5.0 < m < 0.5 and 0.0 < b < 10.0 and -10.0 < log_f < 1.0:
        return 0.0
    return -np.inf

def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)
