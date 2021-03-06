#!/usr/bin/env python

import os
import web
from web import form
import tempfile
import sys
import simplejson as json
import apt
import re
import urllib
import socket
import csv
import datetime
import stat

sys.path.insert(0,"/var/www/AppRecommender/src/")

import logging
from config import Config
from recommender import *
from user import *
from data import DebianPackage

# avoid "RuntimeError: maximum recursion depth exceeded"
sys.setrecursionlimit(50000)

class Index:
    def GET(self):
        return render.index_survey()

class About:
    def GET(self):
        return render.about_survey()

class Thanks:
    def POST(self):
        web_input = web.input()
        user_id = web_input['user_id'].encode('utf8')
        personal_file = open("/var/www/AppRecommender/src/web/submissions/%s/personal" % user_id,'w')
        personal = {}
        for key in ["name","email","user_habits","comments"]:
            if web_input.has_key(key):
                personal[key] = web_input[key].encode("utf-8")
            else:
                personal[key] = ""
        try:
            writer = csv.writer(personal_file)
            writer.writerow(("user","name","email","habits_id","comments"))
            writer.writerow((user_id,personal["name"],personal["email"],
                             personal["user_habits"],personal["comments"]))
        except:
            error_msg = "Could not save optional information."
            logging.critical("Could not save optional information.")
            return render.error([error_msg],"/survey/","START")
        finally:
            personal_file.close()
        return render.thanks_id()

class Fake:
    def GET(self):
        return render_plain.fake()

class Save:
    def POST(self):
        web_input = web.input()
        logging.info("Saving user evaluation...")
        logging.info(web_input)
        user_id = web_input['user_id'].encode('utf8')
        user_dir = "/var/www/AppRecommender/src/web/submissions/%s" % user_id
        strategy = web_input['strategy']
        strategy_dir = os.path.join(user_dir,strategy)
        logging.debug("Saving evaluation for user %s strategy %s."
                      % (user_id,strategy))
        summary = {}
        summary["poor"] = 0
        summary["redundant"] = 0
        summary["useful"] = 0
        summary["surprising"] = 0

        # Save evaluation
        try:
            prediction_file = open(os.path.join(strategy_dir,"prediction"),'r')
            evaluation_file = open(os.path.join(strategy_dir,"evaluation"),'w')
            reader = csv.DictReader(prediction_file)
            writer = csv.DictWriter(evaluation_file,fieldnames=reader.fieldnames)
            headers = dict( (n,n) for n in reader.fieldnames )
            writer.writerow(headers)
            for key, value in web_input.items():
                if key.startswith("evaluation-"):
                    ranking = key.lstrip("evaluation-")
                    for row in reader:
                        if row['ranking'] == ranking:
                            evaluation = value.encode('utf8')
                            row['evaluation'] = evaluation
                            writer.writerow(row)
                            summary[evaluation] += 1
                            break
                    prediction_file.seek(0)
            prediction_file.close()
            evaluation_file.close()
            os.remove(os.path.join(strategy_dir,"prediction"))
        except:
            error_msg = "Could not write evaluation to file."
            logging.critical(error_msg)
            if not os.path.exists(os.path.join(strategy_dir,"evaluation")):
                return render.error([error_msg], "/survey/","START")
        finally:
            with open(os.path.join(strategy_dir,"end"),'w') as end:
                end_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                end.write(end_time)

        # Save report
        try:
            report = os.path.join(user_dir,"report")
            report_file = open(os.path.join(user_dir,"report"),'a')
            writer = csv.writer(report_file)
            if os.path.getsize(report) == 0:
                fieldnames = ('user','strategy','start','end','poor',
                              'redundant','useful','surprising','comments')
                writer.writerow(fieldnames)
            with open(os.path.join(strategy_dir,"start"),'r') as start:
                start_time = start.readline().strip()
            if web_input.has_key("comments"):
                comments =  web_input['comments'].encode("utf-8")
            else:
                comments = ""
            writer.writerow((user_id,strategy,start_time,end_time,summary["poor"],
                             summary["redundant"],summary["useful"],
                             summary["surprising"],comments))
            report_file.close()
        except:
            error_msg = "Could not save evaluation report."
            logging.critical(error_msg)
            if not os.path.exists(os.path.join(user_dir,"report")):
                return render.error([error_msg], "/survey/","START")

        if web_input.has_key('continue_button'):
            return Survey().POST()
        elif web_input.has_key('finish_button'):
            return render.thanks(user_id)
        else:
            return render.index_survey()

class Instruction:
    def POST(self):
        web_input = web.input(pkgs_file={})
        submissions_dir = "/var/www/AppRecommender/src/web/submissions/"
        user_dir = tempfile.mkdtemp(prefix='',dir=submissions_dir)
        os.chmod(user_dir,stat.S_IRWXU|stat.S_IXOTH|stat.S_IROTH)
        user_id = user_dir.split("/")[-1]
        logging.info("Request from user %s" % user_id)
        logging.debug("Created dir %s" % user_dir)
        uploaded_file = os.path.join(user_dir,"uploaded_file")
        if not os.path.exists(uploaded_file):
            if web_input['pkgs_file'].value:
                lines = web_input['pkgs_file'].file.readlines()
                with open(uploaded_file, "w") as uploaded:
                    uploaded.writelines(lines)
            else:
                # saving empty files
                with open(uploaded_file, "w") as uploaded:
                    pass
        return render.intro(user_id)

class Survey:
    def __init__(self):
        logging.info("Setting up survey...")
        self.cfg = Config()
        self.rec = Recommender(self.cfg)
        self.submissions_dir = "/var/www/AppRecommender/src/web/submissions/"
        if not os.path.exists(self.submissions_dir):
            os.makedirs(self.submissions_dir)
        self.strategies = ["cbh","cbh_eset",
                           "knn","knn_eset","knn_plus",
                           "knnco"]
    def POST(self):
        web_input = web.input(pkgs_file={})
        if web_input.has_key('user_id'):
            user_id = web_input['user_id'].encode('utf8')
            user_dir = os.path.join(self.submissions_dir, user_id)
            logging.info("New recommendation for user %s" % user_id)
        
        uploaded_file = os.path.join(user_dir,"uploaded_file")
        with open(uploaded_file) as uploaded:
            if uploaded.readline().startswith('POPULARITY-CONTEST'):
                user = PopconSystem(uploaded_file,user_id)
            else:
                user = PkgsListSystem(uploaded_file,user_id)
        user.maximal_pkg_profile()
        if len(user.pkg_profile)<10:
            error_msg = "Could not extract profile from uploaded file. It must have at least 10 applications."
            logging.critical(error_msg)
            return render.error([error_msg], "/survey/","START")
        else:
            # Check the remaining strategies and select a new one
            old_strategies = [dirs for root, dirs, files in
                              os.walk(os.path.join(self.submissions_dir,
                                                   user_id))]
            if old_strategies:
                strategies = [s for s in self.strategies if s not in old_strategies[0]]
                logging.info("Already used strategies %s" % old_strategies[0])
            else:
                strategies = self.strategies
            if not strategies:
                return render.thanks(user_id)
            selected_strategy = random.choice(strategies)
            logging.info("Selected \'%s\' from %s" % (selected_strategy,strategies))
            self.set_rec_strategy(selected_strategy)
            prediction = self.rec.get_recommendation(user,10).get_prediction()
            logging.info("Prediction for user %s" % user_id)
            logging.info(str(prediction))
            self.save_prediction(user_id,selected_strategy,prediction)

            # Load packages details
            recommendation = [result[0] for result in prediction]
            pkgs_details = []
            for pkg_name in recommendation:
                logging.info("Getting details of package %s" % pkg_name)
                pkg = DebianPackage(pkg_name)
                pkg.load_details()
                pkgs_details.append(pkg)

            if pkgs_details:
                logging.info("Rendering survey slide...")
                return render.survey(pkgs_details, user_id, selected_strategy, len(strategies))
            else:
                return render.error(["No recommendation produced for the uploaded file."],"/survey/","START")

    def set_rec_strategy(self,selected_strategy):
        k=10
        n=20
        if selected_strategy == "cbh":
            pass
        if selected_strategy == "cbh_eset":
            pass
        if selected_strategy == "knn":
            pass
        if selected_strategy == "knn_eset":
            pass
        if selected_strategy == "knn_plus":
            pass
        if selected_strategy == "knnco":
            pass
        self.rec.set_strategy(selected_strategy,k,n)
        return selected_strategy

    def save_prediction(self,user_id,strategy,prediction):
        strategy_dir = os.path.join(self.submissions_dir,user_id,strategy)
        if not os.path.exists(strategy_dir):
            os.makedirs(strategy_dir)
        ranking = 0
        prediction_file = open(os.path.join(strategy_dir,"prediction"),"w")
        try:
            writer = csv.writer(prediction_file)
            fieldnames = ('ranking','rating','package','evaluation')
            writer.writerow(fieldnames)
            for pkg,rating in prediction:
                writer.writerow((ranking,"%.4f"%rating,pkg,""))
                ranking += 1
        except:
            error_msg = "Error to write prediction to file."
            logging.critical(error_msg)
            return render.error([error_msg], "/survey/","START")
        finally:
            prediction_file.close()
        with open(os.path.join(strategy_dir,"start"),'w') as start:
            now = datetime.datetime.now()
            start.write(now.strftime("%Y%m%d%H%M%S"))
        logging.debug("Saved prediction to file at %s/%s" %
                      (user_id,strategy))
#def add_global_hook():
#    g = web.storage({"counter": "1"})
#    def _wrapper(handler):
#        web.ctx.globals = g
#        return handler()
#    return _wrapper

render = web.template.render('/var/www/AppRecommender/src/web/templates/', base='layout', globals={'hasattr':hasattr})
render_plain = web.template.render('/var/www/AppRecommender/src/web/templates/', globals={'hasattr':hasattr})

urls = ('/apprec',         'Instruction',
        '/evaluation',     'Survey',
        '/thanks',   	   'Thanks',
        '/save',   	   'Save',
        '/about',          'About',
        '/index',          'Index',
        '/', 	           'Index',
#        '/', 	           'Fake',
       )

web.webapi.internalerror = web.debugerror

cfg = Config()
app = web.application(urls, globals(), autoreload=False)
application = app.wsgifunc()
