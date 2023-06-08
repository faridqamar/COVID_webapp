import os
from app import app
from app import r
from app import q
from app.tasks import send_file_to_remote, extract_results, file_cleanup, test_email, quick_plot_remote
from time import strftime
from datetime import datetime
from flask import render_template
from flask import request, redirect, url_for
from flask import jsonify, make_response
from flask import flash
from flask import abort
from flask_mail import Mail, Message
from werkzeug.utils import secure_filename
from werkzeug.datastructures import ImmutableOrderedMultiDict
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import corner
import csv

# CHANGE WHEN OUT OF DEVELOPMENT
app.config['FLASK_ENV'] = 'development'
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
#app.config['MAIL_PORT'] = 587
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = '' # sender email address
app.config['MAIL_PASSWORD'] = '' # sender email password
#app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
#app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False

app.config["CSV_UPLOADS"] = "../../covid_dev/app/static/csv/uploads"
app.config["CSV_CONFIGS"] = "../../covid_dev/app/static/csv/configs"
app.config["ALLOWED_CSV_EXTENSIONS"] = ["CSV"]
#app.config["RESULTS_DIRECTORY"] = "../../covid_dev/app/static/csv/outputs/"
app.config["RESULTS_DIRECTORY"] = "../../covid_dev/app/static/results/"
app.config["QUICK_PLOT_DIR"] = "../../covid_dev/app/static/img/output/"
app.config["MAX_CSV_FILESIZE"] = 10.0 * 1024 * 1024 #10.0MB
app._static_folder = "../../covid_dev/app/static/"

mail = Mail(app)

def allowed_file_ext(filename):
    if not "." in filename:
        return False
    ext = filename.rsplit(".")
    
    if ext[-1].upper() in app.config["ALLOWED_CSV_EXTENSIONS"]:
        return True
    else:
        return False

def allowed_file_filesize(filesize):

    if int(filesize) <= app.config["MAX_CSV_FILESIZE"]:
        return True
    else:
        return False

def allowed_file_formatting(df):
    # file must have 3 columns
#    if len(df.keys()) != 3:
#        return False
    
    # column 1 must be labeled x, column 2 labeled y, column 3 labeled yerr
#    if df.keys()[0] != 'x' or df.keys()[1] != 'y' or df.keys()[2] != 'yerr':
#        return False
    
    # columns must have same length, and only contain numbers
#    lenx = len(df[0])
#    for key in df.keys():
#        if len(df[key]) != lenx:
#            return False
#        for ii in df[key]:
#            if type(ii) != int:
#                if type(ii) != float:
#                    return False
    
    return True

now = datetime.now()
today = now.strftime('%Y-%m-%d')

# list of all emails authorized to use the application
auth_emails = []

@app.route("/", methods=["GET", "POST"])
def index():

    # start wit file cleanup
#    file_cleanup(app.config["CSV_UPLOADS"])
#    file_cleanup(app.config["RESULTS_DIRECTORY"])

    if request.method == "POST":
        if request.files:
            # check user email is in list
            email = request.form["email"]
            if email.lower() not in auth_emails:
                flash("Email address is not authenticated, contact the administrator to gain access.", "warning")
                return redirect(request.url)

            upfile = request.files["upfile"]
            
            # check a file was chosen (not necessary with required=true in html)
            if upfile.filename == "":
                flash("No file selected! Select .csv file to upload.", "warning")
                return redirect(request.url)
            
            # check extension is allowed 
            if not allowed_file_ext(upfile.filename):
                flash("File extension not allowed! Select .csv file to upload.", "warning")
                return redirect(request.url)
            
            # check filesize does not exceed maximum limit
            if not allowed_file_filesize(request.cookies["filesize"]):
                flash("File not uploaded! Filesize exceeded maximum limit", "warning")
                return redirect(request.url)
            
            # all's well, save file
            #filename = secure_filename(upfile.filename)
            timestamp = str(strftime("%Y%m%d_%H%M%S"))
            randstamp = str(random.randint(0, 100000))
            filecode = str(timestamp + "_" + randstamp)
            filename = str("uploaded_data_file_" + filecode + ".csv")
            upfile.save(os.path.join(app.config["CSV_UPLOADS"], filename))
            
            filepath = str(app.config["CSV_UPLOADS"] + "/" + filename)
            df = pd.read_csv(filepath)

            # check file structure and formatting
            if not allowed_file_formatting(df):
                flash("Incompatible file structure! See help page for instructions on formatting your csv file.", "warning")
                return redirect(request.url)
            

            # all's well, continue getting variables
            DEorCC = request.form.get("DEorCC")
            npop = request.form.get("npop")
            MCMC_nsteps = request.form.get("MCMC_nsteps")
            if MCMC_nsteps == "Low":
                nsteps = 600
            elif MCMC_nsteps == "Med":
                nsteps = 2000
            else:
                nsteps = 10000

            #npop = request.form.get("npop")
            fracwith = request.form.get("fracwith")
            daysuntils = request.form.get("daysuntils")
            infectious_days = request.form.get("infectious_days")
            symptoms_days_before_hospital = request.form.get("symptoms_days_before_hospital")

            checkreinsert = request.form.get("checkreinsert")
            reinsert = np.inf
            if checkreinsert == "on":
                reinsert = request.form.get("reinsert")
                if reinsert == "": reinsert = "inf"

            checkvax = request.form.get("checkvax")
            vaccinated = np.inf
            if checkvax == "on":
                vaccinated = request.form.get("vaccinated")
                if vaccinated == "": vaccinated = "inf"

            checkdates = request.form.get("checkdates")
            DistancingDates = []
            DistancingDurations = []
            DistancingExtent = []
            if checkdates == "on":
                DistancingDates = request.form.getlist("dates")
                DistancingDurations = request.form.getlist("durations")
                DistancingExtent = request.form.getlist("extent")

            username = request.form["username"]
            email = request.form["email"]

        #    LS_MS_mcmc_fits(username, email, df, nsteps)
        #    send_file_to_remote(filename, nsteps, email, username)
            #message=f'task successful look for {filename} in input directory'

            print(username, email, DEorCC, npop, nsteps, fracwith, daysuntils, infectious_days, 
                  symptoms_days_before_hospital, checkreinsert, reinsert,
                  checkvax, vaccinated, checkdates, DistancingDates, DistancingDurations, DistancingExtent)

            jobs = q.jobs  # Get a list of jobs in the queue
            task = q.enqueue(send_file_to_remote, args=(DEorCC, npop, nsteps, fracwith, daysuntils, infectious_days,
                                                         symptoms_days_before_hospital, reinsert, vaccinated,
                                                         DistancingDates, DistancingDurations, DistancingExtent,
                                                         filename, filecode, email, username), timeout=18000)  # Send a job to the task queue (timeout = 5hours)
            jobs = q.jobs  # Get a list of jobs in the queue
            q_len = len(q)  # Get the queue length
            message = f"Task queued at {task.enqueued_at.strftime('%a, %d %b %Y %H:%M:%S')}. {q_len} other jobs queued"

            return render_template("public/jobsubmit.html",
                                   username=username, email=email, message=message, jobs=jobs,
                                   checkdates=checkdates, DistancingDates=DistancingDates,
                                   DistancingDurations=DistancingDurations)

#            mm = test_email(email)
#            return render_template("public/index.html", today=today, auth_emails=auth_emails)

    return render_template("public/index.html", today=today, auth_emails=auth_emails)


@app.route("/plotting", methods=["GET", "POST"])
def plotting():
    plotname = ''
    return render_template("public/plotting.html", today=today, plotname=plotname)

@app.route("/quickplot", methods=["GET", "POST"])
def quickplot():
    # start wit file cleanup
#    file_cleanup(app.config["CSV_UPLOADS"])
#    file_cleanup(app.config["RESULTS_DIRECTORY"])
    plotname = ''
    if request.method == "POST":
       if request.files:
            upfile = request.files["upfile"]
            # check a file was chosen (not necessary with required=true in html)
            if upfile.filename == "":
                flash("No file selected! Select .csv file to upload.", "warning")
                return redirect(request.url)

            # check extension is allowed 
            if not allowed_file_ext(upfile.filename):
                flash("File extension not allowed! Select .csv file to upload.", "warning")
                return redirect(request.url)
            
            # check filesize does not exceed maximum limit
            if not allowed_file_filesize(request.cookies["filesize"]):
                flash("File not uploaded! Filesize exceeded maximum limit", "warning")
                return redirect(request.url)

            # all's well, save file
            #filename = secure_filename(upfile.filename)
            timestamp = str(strftime("%Y%m%d_%H%M%S"))
            randstamp = str(random.randint(0, 100000))
            filecode = str(timestamp + "_" + randstamp)
            filename = str("uploaded_data_file_" + filecode + ".csv")
            upfile.save(os.path.join(app.config["CSV_UPLOADS"], filename))
            
            filepath = str(app.config["CSV_UPLOADS"] + "/" + filename)
            df = pd.read_csv(filepath)

            # check file structure and formatting
            if not allowed_file_formatting(df):
                flash("Incompatible file structure! See help page for instructions on formatting your csv file.", "warning")
                return redirect(request.url)
            

            # all's well, continue getting variables
            DEorCC = request.form.get("DEorCC")
            npop = request.form.get("npop")
            nsteps = 700
            #npop = request.form.get("npop")
            fracwith = request.form.get("fracwith")
            daysuntils = request.form.get("daysuntils")
            infectious_days = request.form.get("infectious_days")
            symptoms_days_before_hospital = request.form.get("symptoms_days_before_hospital")

            checkreinsert = request.form.get("checkreinsert")
            reinsert = np.inf
            if checkreinsert == "on":
                reinsert = request.form.get("reinsert")
                if reinsert == "": reinsert = "inf"

            checkvax = request.form.get("checkvax")
            vaccinated = np.inf
            if checkvax == "on":
                vaccinated = request.form.get("vaccinated")
                if vaccinated == "": vaccinated = "inf"

            checkdates = request.form.get("checkdates")
            DistancingDates = []
            DistancingDurations = []
            DistancingExtent = []
            if checkdates == "on":
                DistancingDates = request.form.getlist("dates")
                DistancingDurations = request.form.getlist("durations")
                DistancingExtent = request.form.getlist("extent")

            plotname = quick_plot_remote(DEorCC, npop, nsteps, fracwith, daysuntils, infectious_days,
                                         symptoms_days_before_hospital, reinsert, vaccinated,
                                         DistancingDates, DistancingDurations, DistancingExtent,
                                         filename, filecode)
#            print(plotname)
#            return jsonify({'output': plotname})
            
            return jsonify(plotname)
#            return render_template("public/plotting.html", today=today, plotname=plotname)

#    return render_template("public/plotting.html", today=today, plotname=plotname)



app.config["RES_UPLOADS"] = "../../covid_dev/app/static/csv/uploads"
app.config["PLOT_RESULTS"] = "../../covid_dev/app/static/img/output"
@app.route("/output-result", methods=["GET", "POST"])
def output_result():

    # start with file cleanup
    file_cleanup(app.config["RES_UPLOADS"])
    file_cleanup(app.config["PLOT_RESULTS"])

    if request.method == "POST":
        if request.files:
            resfile = request.files["resfile"]
            
            # check a file was chosen (not necessary with required=true in html)
            if resfile.filename == "":
                flash("No file selected! Upload the file attached in the email sent to you.", "warning")
                return redirect(request.url)
            
            # check extension is allowed
            if not allowed_file_ext(resfile.filename):
                flash("File extension not allowed! Upload the file attached in the email sent to you.", "warning")
                return redirect(request.url)
            
            # check filesize does not exceed maximum limit
            if not allowed_file_filesize(request.cookies["filesize"]):
                flash("File not uploaded! Filesize exceeded maximum limit", "warning")
                return redirect(request.url)
            
            # all's well, save file
            #filename = secure_filename(upfile.filename)
            timestamp = str(strftime("%Y%m%d_%H%M%S"))
            randstamp = str(random.randint(0, 100000))
            filecode = str(timestamp + "_" + randstamp)
            filename = str("uploaded_results_file_" + filecode + ".csv")
            filepath = os.path.join(app.config["RES_UPLOADS"], filename)
            resfile.save(filepath)
                
            print("Results file saved")
                
            print("Reading results csv file...")
            
            try:
                plotname, ls_soln, ml_soln, nsteps, mcmc_m, mcmc_b, mcmc_f, df = extract_results(filepath, filecode)
                
                #labels = ["m", "b", "log(f)"]
                #fig = corner.corner(
                #mcmc_flat_samples, labels=labels)
                #fig.savefig(os.path.join(tardir,'corner.png'), bbox_inches='tight')
                #plt.close('all')

                print("sending variables to results page")
                return render_template("public/results.html", plotname=plotname, ls_soln=ls_soln, ml_soln=ml_soln, 
                                        nsteps=nsteps, mcmc_m=mcmc_m, mcmc_b=mcmc_b, mcmc_f=mcmc_f, df=df)
                #return render_template("public/results.html", plotname=plotname, ls_soln=ls_soln, ml_soln=ml_soln, df=df)
            
            except:
                flash("Incompatible file! Upload the file attached in the email sent to you.", "warning")
                return redirect(request.url)
            
    return render_template("public/output_result.html")


@app.route("/results/<filecode>")
def results(filecode):
    #filename = str("results_" + filecode + ".csv")
    #filepath = os.path.join(app.config["RESULTS_DIRECTORY"], filename)

    resdir = os.path.join(app.config["RESULTS_DIRECTORY"], filecode)

    if os.path.isdir(resdir):
        parameters = pd.read_csv(os.path.join(resdir, 'parameters.csv'))
        return render_template("public/results.html", filecode=filecode, parameters=parameters)

#    if os.path.isfile(filepath):
#        plotname, ls_soln, ml_soln, nsteps, mcmc_m, mcmc_b, mcmc_f, df = extract_results(filepath, filecode)
#        return render_template("public/results.html", plotname=plotname, ls_soln=ls_soln, ml_soln=ml_soln,
#                                nsteps=nsteps, mcmc_m=mcmc_m, mcmc_b=mcmc_b, mcmc_f=mcmc_f, df=df)
    
    return render_template("public/error_page.html")


@app.route("/tasks", methods=["GET", "POST"])
def tasks():
    jobs = q.jobs  # Get a list of jobs in the queue
    q_len = len(q)  # Get the queue length

    return render_template("public/tasks.html", jobs=jobs)


# error handling
#@app.errorhandler(403)
#def forbidden(e):
#    return render_template("error_handlers/forbidden.html"), 403

#@app.errorhandler(404)
#def page_not_found(e):
#    app.logger.info(f"Page not found: {request.url}")

#    return render_template("error_handlers/404.html"), 404

#@app.errorhandler(500)
#def server_error(e):
#    email_admin(message="Server error", url=request.url, error=e)
#    app.logger.error(f"Server error: {request.url}")

#    return render_template("error_handlers/500.html"), 500


@app.template_filter("clean_date")
def clean_date(dt):
    return dt.strftime("%d %b %Y")

@app.template_filter("round_var")
def round_var(var):
    return np.round(var, 2)
