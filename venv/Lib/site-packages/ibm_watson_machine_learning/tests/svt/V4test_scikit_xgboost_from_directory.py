import unittest
from sklearn.datasets import load_svmlight_file
import logging
import datetime
from ibm_watson_machine_learning.tests.utils import get_wml_credentials, get_cos_credentials, get_space_id
from ibm_watson_machine_learning.tests.utils.cleanup import space_cleanup
from ibm_watson_machine_learning import APIClient
from models_preparation import *


class TestWMLClientWithXGBoost(unittest.TestCase):
    deployment_uid = None
    model_uid = None
    scoring_url = None
    space_name = 'tests_sdk_space'
    space_id = None
    model_path = os.path.join('.', 'svt', 'artifacts', 'scikit_xgboost_model_' + datetime.datetime.now().isoformat())
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(cls) -> None:
        """
        Load WML credentials from config.ini file based on ENV variable.
        """

        cls.wml_credentials = get_wml_credentials()

        cls.wml_client = APIClient(wml_credentials=cls.wml_credentials)

        if not cls.wml_client.ICP:
            cls.cos_credentials = get_cos_credentials()
            cls.cos_endpoint = cls.cos_credentials.get('endpoint_url')
            cls.cos_resource_instance_id = cls.cos_credentials.get('resource_instance_id')

        cls.wml_client = APIClient(wml_credentials=cls.wml_credentials)
        cls.project_id = cls.wml_credentials.get('project_id')

    def test_00a_space_cleanup(self):
        space_cleanup(self.wml_client,
                      get_space_id(self.wml_client, self.space_name,
                                   cos_resource_instance_id=self.cos_resource_instance_id),
                      days_old=7)
        TestWMLClientWithXGBoost.space_id = get_space_id(self.wml_client, self.space_name,
                                                         cos_resource_instance_id=self.cos_resource_instance_id)

        # if self.wml_client.ICP:
        #     self.wml_client.set.default_project(self.project_id)
        # else:
        self.wml_client.set.default_space(self.space_id)

    def test_2_publish_model(self):
        TestWMLClientWithXGBoost.logger.info("Publishing scikit-xgboost model ...")

        import shutil

        try:
            shutil.rmtree(self.model_path)
        except:
            pass

        create_scikit_learn_xgboost_model_directory(self.model_path)

        self.wml_client.repository.ModelMetaNames.show()

        self.wml_client.software_specifications.list()
        sw_spec_id = self.wml_client.software_specifications.get_id_by_name("scikit-learn_0.20-py3.6")

        model_props = {self.wml_client.repository.ModelMetaNames.NAME: "scikit_xg",
            self.wml_client.repository.ModelMetaNames.TYPE: "scikit-learn_0.20",
            self.wml_client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: sw_spec_id
                       }
        published_model = self.wml_client.repository.store_model(model=self.model_path, meta_props=model_props)
        TestWMLClientWithXGBoost.model_uid = self.wml_client.repository.get_model_uid(published_model)
        TestWMLClientWithXGBoost.logger.info("Published model ID:" + str(TestWMLClientWithXGBoost.model_uid))
        self.assertIsNotNone(TestWMLClientWithXGBoost.model_uid)

        shutil.rmtree(self.model_path)

    def test_3_get_details(self):
        TestWMLClientWithXGBoost.logger.info("Get published model details ...")
        details = self.wml_client.repository.get_details(self.model_uid)

        TestWMLClientWithXGBoost.logger.debug("Model details: " + str(details))
        self.assertTrue("scikit_xg" in str(details))

    def test_4_create_deployment(self):
        TestWMLClientWithXGBoost.logger.info("Create deployment ...")
        global deployment
        deployment = self.wml_client.deployments.create(self.model_uid, meta_props={self.wml_client.deployments.ConfigurationMetaNames.NAME: "Test deployment",self.wml_client.deployments.ConfigurationMetaNames.ONLINE:{}})

        TestWMLClientWithXGBoost.logger.info("model_uid: " + self.model_uid)
        TestWMLClientWithXGBoost.logger.info("Online deployment: " + str(deployment))
        TestWMLClientWithXGBoost.scoring_url = self.wml_client.deployments.get_scoring_href(deployment)
        self.assertTrue("online" in str(deployment))

    def test_5_get_deployment_details(self):
        TestWMLClientWithXGBoost.logger.info("Get deployment details ...")
        deployment_details = self.wml_client.deployments.get_details()
        self.assertTrue("Test deployment" in str(deployment_details))

        TestWMLClientWithXGBoost.logger.debug("Online deployment: " + str(deployment_details))
        TestWMLClientWithXGBoost.deployment_uid = self.wml_client.deployments.get_uid(deployment)
        self.assertIsNotNone(TestWMLClientWithXGBoost.deployment_uid)

    def test_6_score(self):
        TestWMLClientWithXGBoost.logger.info("Online model scoring ...")
        (X, _) = load_svmlight_file(os.path.join('svt', 'artifacts', 'agaricus.txt.test'))
        scoring_data = {
            self.wml_client.deployments.ScoringMetaNames.INPUT_DATA: [
                {
                    'values': [list(X.toarray()[0, :])]
                }
            ]
        }
        TestWMLClientWithXGBoost.logger.debug("Scoring data: {}".format(scoring_data))
        predictions = self.wml_client.deployments.score(TestWMLClientWithXGBoost.deployment_uid, scoring_data)

        TestWMLClientWithXGBoost.logger.debug("Prediction: " + str(predictions))
        self.assertTrue(("prediction" in str(predictions)) and ("probability" in str(predictions)))

    def test_7_delete_deployment(self):
        TestWMLClientWithXGBoost.logger.info("Delete model deployment ...")
        self.wml_client.deployments.delete(TestWMLClientWithXGBoost.deployment_uid)

    def test_8_delete_model(self):
        TestWMLClientWithXGBoost.logger.info("Delete model ...")
        self.wml_client.repository.delete(TestWMLClientWithXGBoost.model_uid)


if __name__ == '__main__':
    unittest.main()
