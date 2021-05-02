import unittest
import datetime
import logging
from sklearn import datasets
from ibm_watson_machine_learning.wml_client_error import WMLClientError
from ibm_watson_machine_learning.tests.utils import get_wml_credentials, get_cos_credentials, get_space_id
from ibm_watson_machine_learning.tests.utils.cleanup import space_cleanup
from ibm_watson_machine_learning import APIClient
from models_preparation import *


class TestWMLClientWithScikitLearn(unittest.TestCase):
    deployment_uid = None
    model_uid = None
    scoring_url = None
    space_name = 'tests_sdk_space'
    space_id = None
    model_path = os.path.join('.', 'svt', 'artifacts', 'scikit_model_' + datetime.datetime.now().isoformat())
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
        TestWMLClientWithScikitLearn.space_id = get_space_id(self.wml_client, self.space_name,
                                                       cos_resource_instance_id=self.cos_resource_instance_id)

        # if self.wml_client.ICP:
        #     self.wml_client.set.default_project(self.project_id)
        # else:
        self.wml_client.set.default_space(self.space_id)

    def test_02_publish_model(self):
        TestWMLClientWithScikitLearn.logger.info("Publish model")
        global digits
        digits = datasets.load_digits()

        import shutil

        try:
            shutil.rmtree(self.model_path)
        except:
            pass

        create_scikit_learn_model_directory(self.model_path)

        self.logger.info("Publishing scikit-learn model ...")

        self.wml_client.repository.ModelMetaNames.show()

        sw_spec_id = self.wml_client.software_specifications.get_id_by_name("scikit-learn_0.20-py3.6")

        model_props = {self.wml_client.repository.ModelMetaNames.NAME: "LOCALLY created Digits prediction model",
            self.wml_client.repository.ModelMetaNames.TYPE: "scikit-learn_0.20",
            self.wml_client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: sw_spec_id
                       }
        published_model_details = self.wml_client.repository.store_model(model=self.model_path, meta_props=model_props, training_data=digits.data, training_target=digits.target)
        TestWMLClientWithScikitLearn.model_uid = self.wml_client.repository.get_model_uid(published_model_details)
        TestWMLClientWithScikitLearn.model_url = self.wml_client.repository.get_model_href(published_model_details)
        TestWMLClientWithScikitLearn.logger.info("Published model ID:" + str(TestWMLClientWithScikitLearn.model_uid))
        TestWMLClientWithScikitLearn.logger.info("Published model URL:" + str(TestWMLClientWithScikitLearn.model_url))
        self.assertIsNotNone(TestWMLClientWithScikitLearn.model_uid)
        self.assertIsNotNone(TestWMLClientWithScikitLearn.model_url)

        shutil.rmtree(self.model_path)

    def test_03_download_model(self):
        TestWMLClientWithScikitLearn.logger.info("Download model")
        try:
            os.remove('download_test_url')
        except OSError:
            pass

        try:
            file = open('download_test_uid', 'r')
        except IOError:
            file = open('download_test_uid', 'w')
            file.close()

        self.wml_client.repository.download(TestWMLClientWithScikitLearn.model_uid, filename='download_test_url')
        self.assertRaises(WMLClientError, self.wml_client.repository.download, TestWMLClientWithScikitLearn.model_uid, filename='download_test_uid')

    def test_04_get_details(self):
        TestWMLClientWithScikitLearn.logger.info("Get model details")

        details = self.wml_client.repository.get_details(self.model_uid)
        TestWMLClientWithScikitLearn.logger.debug("Model details: " + str(details))
        self.assertTrue("LOCALLY created Digits prediction model" in str(details))

        details_all = self.wml_client.repository.get_details()
        TestWMLClientWithScikitLearn.logger.debug("All artifacts details: " + str(details_all))
        self.assertTrue("LOCALLY created Digits prediction model" in str(details_all))

        details_models = self.wml_client.repository.get_model_details()
        TestWMLClientWithScikitLearn.logger.debug("All models details: " + str(details_models))
        self.assertTrue("LOCALLY created Digits prediction model" in str(details_models))

    def test_05_create_deployment(self):
        TestWMLClientWithScikitLearn.logger.info("Create deployment")
        deployment = self.wml_client.deployments.create(self.model_uid, meta_props={
            self.wml_client.deployments.ConfigurationMetaNames.NAME: "Test deployment",
            self.wml_client.deployments.ConfigurationMetaNames.ONLINE: {}})


        TestWMLClientWithScikitLearn.logger.info("model_uid: " + self.model_uid)
        TestWMLClientWithScikitLearn.logger.info("Online deployment: " + str(deployment))
        self.assertTrue(deployment is not None)
        TestWMLClientWithScikitLearn.scoring_url = self.wml_client.deployments.get_scoring_href(deployment)
        TestWMLClientWithScikitLearn.deployment_uid = self.wml_client.deployments.get_uid(deployment)
        self.assertTrue("online" in str(deployment))

    def test_06_get_deployment_details(self):
        TestWMLClientWithScikitLearn.logger.info("Get deployment details")

        deployment_details = self.wml_client.deployments.get_details()
        print(deployment_details)
        self.assertTrue("Test deployment" in str(deployment_details))

    def test_07_get_deployment_details_using_uid(self):
        TestWMLClientWithScikitLearn.logger.info("Get deployment details using uid")
        deployment_details = self.wml_client.deployments.get_details(TestWMLClientWithScikitLearn.deployment_uid)
        self.assertIsNotNone(deployment_details)

    def test_08_score(self):
        TestWMLClientWithScikitLearn.logger.info("Score model")
        scoring_data = {
            self.wml_client.deployments.ScoringMetaNames.INPUT_DATA: [
                {
                    'values': [
                        [0.0, 0.0, 5.0, 16.0, 16.0, 3.0, 0.0, 0.0, 0.0, 0.0, 9.0, 16.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         12.0, 15.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 15.0, 16.0, 15.0, 4.0, 0.0, 0.0, 0.0, 0.0, 9.0, 13.0,
                         16.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 14.0, 12.0, 0.0, 0.0, 0.0, 0.0, 5.0, 12.0, 16.0, 8.0,
                         0.0, 0.0, 0.0, 0.0, 3.0, 15.0, 15.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 6.0, 16.0, 12.0, 1.0, 0.0, 0.0, 0.0, 0.0, 5.0, 16.0, 13.0, 10.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 5.0, 5.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         13.0, 13.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 16.0, 9.0, 4.0, 1.0, 0.0, 0.0, 3.0, 16.0, 16.0, 16.0,
                         16.0, 10.0, 0.0, 0.0, 5.0, 16.0, 11.0, 9.0, 6.0, 2.0]]
                }
            ]
        }
        predictions = self.wml_client.deployments.score(TestWMLClientWithScikitLearn.deployment_uid, scoring_data)
        self.assertTrue("prediction" in str(predictions))

    def test_09_delete_deployment(self):
        TestWMLClientWithScikitLearn.logger.info("Delete deployment")
        self.wml_client.deployments.delete(TestWMLClientWithScikitLearn.deployment_uid)

    def test_10_delete_model(self):
        TestWMLClientWithScikitLearn.logger.info("Delete model")
        self.wml_client.repository.delete(TestWMLClientWithScikitLearn.model_uid)


if __name__ == '__main__':
    unittest.main()
