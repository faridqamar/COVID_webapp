import time
import boto3
import paramiko
from botocore.exceptions import ClientError
import email, smtplib, ssl
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

ec2Resource = boto3.resource('ec2')
ec2Client = boto3.client('ec2')

pltInstID = 'i-' # insert plotting instance id
cmpInstID = 'i-' # insert computing instance id

plt_remote_instance = ec2Resource.Instance(pltInstID)
cmp_remote_instance = ec2Resource.Instance(cmpInstID)

plt_instance_status = plt_remote_instance.state['Name']
cmp_instance_status = cmp_remote_instance.state['Name']

#print('Plotting instance is ' + plt_instance_status)
#print('Compute instance is ' + cmp_instance_status)

plt_error = 'instance already running'
cmp_error = 'instance already running'

if plt_instance_status == 'stopped':
#    print('Starting Plotting instance')
    try:
        ec2Client.start_instances(InstanceIds=[pltInstID])
        plt_error = 'instance started'
    except ClientError as e:
#        print(e)
        plt_error = str(e)

if cmp_instance_status == 'stopped':
#    print('Starting Compute instance')
    try:
        ec2Client.start_instances(InstanceIds=[cmpInstID])
        cmp_error = 'instance started'
    except ClientError as e:
#        print(e)
        cmp_error = str(e)

time.sleep(60)

plt_instance_status = plt_remote_instance.state['Name']
cmp_instance_status = cmp_remote_instance.state['Name']

#print('Plotting instance is ' + plt_instance_status)
#print('Compute instance is ' + cmp_instance_status)

sender_email = '' # email address to send from
password = ''  # password for the sender email address
receiver_email = '' # email address of the recipient

subject = "COVID instance startup"

body = f"""
    Plotting Instance:
        Status: {plt_instance_status}
        Errors: {plt_error}

    Compute Instance:
        Status: {cmp_instance_status}
        Errors: {cmp_error}


    """

message = MIMEMultipart()
message["From"] = sender_email
message["To"] = receiver_email
message["Subject"] = subject

message.attach(MIMEText(body, "plain"))

text = message.as_string()

context = ssl.create_default_context()

with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as server:
    server.login(sender_email, password)
    server.sendmail(
        sender_email, receiver_email, text
    )
