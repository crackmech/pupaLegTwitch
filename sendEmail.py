#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 00:06:02 2015

@author: pointgrey
"""

#http://naelshiab.com/tutorial-send-email-python/

import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText
import re
import numpy as np
import random
import time
from datetime import datetime
import sys


def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def present_time():
        return datetime.now().strftime('%Y%m%d_%H%M%S')


def mail(to, subject, text, user, password):
   msg = MIMEMultipart()
   msg['From'] = "crack.mech@gmail.com"
   msg['To'] = to
   msg['Subject'] = subject
   msg.attach(MIMEText(text))
   mailServer = smtplib.SMTP("smtp.gmail.com", 587)
   mailServer.ehlo()
   mailServer.starttls()
   mailServer.ehlo()
   mailServer.login(user, password)
   mailServer.sendmail(user, to, msg.as_string())
   # Should be mailServer.quit(), but that crashes...
   mailServer.close()


gmail_user = "crack.mech@gmail.com"
gmail_pwd = "Amuwal33"

sendTo = 'crack.mech@gmail.com'
subject = "Imaging Crashed!!! on " + present_time()
body = 'The imaging on pointgrey computer has stopped/crashed on '+present_time()

print 'Sending email to: '+sendTo
mail(sendTo, subject, body, gmail_user, gmail_pwd)
print 'Sent mail on '+present_time()












