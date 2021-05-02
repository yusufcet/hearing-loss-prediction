import unittest
import logging
import os
from ibm_watson_machine_learning.tests.utils import get_wml_credentials, get_cos_credentials, get_space_id
from ibm_watson_machine_learning.tests.utils.cleanup import space_cleanup
from ibm_watson_machine_learning import APIClient


class TestWMLClientWithSPSS(unittest.TestCase):
    deployment_uid = None
    model_uid = None
    scoring_url = None
    scoring_uid = None
    space_id = None
    space_name = 'tests_sdk_space'
    model_path = os.path.join('.', 'svt', 'artifacts', 'customer-satisfaction-prediction.str')
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
        TestWMLClientWithSPSS.space_id = get_space_id(self.wml_client, self.space_name,
                                                       cos_resource_instance_id=self.cos_resource_instance_id)

        # if self.wml_client.ICP:
        #     self.wml_client.set.default_project(self.project_id)
        # else:
        self.wml_client.set.default_space(self.space_id)

    def test_02_publish_local_model_in_repository(self):
        TestWMLClientWithSPSS.logger.info("Saving trained model in repo ...")
        TestWMLClientWithSPSS.logger.debug("Model path: {}".format(self.model_path))

        self.wml_client.repository.ModelMetaNames.show()

        self.wml_client.software_specifications.list()
        sw_spec_id = self.wml_client.software_specifications.get_id_by_name("spss-modeler_18.1")

        model_meta_props = {self.wml_client.repository.ModelMetaNames.NAME: "LOCALLY created agaricus prediction model",
                       self.wml_client.repository.ModelMetaNames.TYPE: "spss-modeler_18.1",
                       self.wml_client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: sw_spec_id
                            }
        published_model = self.wml_client.repository.store_model(model=self.model_path, meta_props=model_meta_props)
        TestWMLClientWithSPSS.model_uid = self.wml_client.repository.get_model_uid(published_model)
        TestWMLClientWithSPSS.logger.info("Published model ID:" + str(TestWMLClientWithSPSS.model_uid))
        self.assertIsNotNone(TestWMLClientWithSPSS.model_uid)

    def test_03_load_model(self):
        TestWMLClientWithSPSS.logger.info("Load model from repository: {}".format(TestWMLClientWithSPSS.model_uid))
        self.tf_model = self.wml_client.repository.load(TestWMLClientWithSPSS.model_uid)
        TestWMLClientWithSPSS.logger.debug("SPSS type: {}".format(type(self.tf_model)))
        self.assertTrue(self.tf_model)

    def test_04_get_details(self):
        TestWMLClientWithSPSS.logger.info("Get details")
        self.assertIsNotNone(self.wml_client.repository.get_details(TestWMLClientWithSPSS.model_uid))

    def test_05_get_model_details(self):
        TestWMLClientWithSPSS.logger.info("Get model details")
        self.assertIsNotNone(self.wml_client.repository.get_model_details(TestWMLClientWithSPSS.model_uid))

    def test_07_create_deployment(self):
        TestWMLClientWithSPSS.logger.info("Create deployment")
        deployment_details = self.wml_client.deployments.create(TestWMLClientWithSPSS.model_uid, meta_props={self.wml_client.deployments.ConfigurationMetaNames.NAME: "Test deployment",self.wml_client.deployments.ConfigurationMetaNames.ONLINE:{}})

        TestWMLClientWithSPSS.logger.debug("Deployment details: {}".format(deployment_details))

        TestWMLClientWithSPSS.deployment_uid = self.wml_client.deployments.get_uid(deployment_details)
        TestWMLClientWithSPSS.logger.debug("Deployment uid: {}".format(TestWMLClientWithSPSS.deployment_uid))

        TestWMLClientWithSPSS.scoring_url = self.wml_client.deployments.get_scoring_href(deployment_details)
        TestWMLClientWithSPSS.logger.debug("Scoring url: {}".format(TestWMLClientWithSPSS.scoring_url))

        self.assertTrue('online' in str(deployment_details))

    def test_08_get_deployment_details(self):
        TestWMLClientWithSPSS.logger.info("Get deployment details")
        deployment_details = self.wml_client.deployments.get_details(deployment_uid=TestWMLClientWithSPSS.deployment_uid)
        self.assertIsNotNone(deployment_details)

    def test_09_scoring(self):
        TestWMLClientWithSPSS.logger.info("Score the model")
        scoring_payload = {
            self.wml_client.deployments.ScoringMetaNames.INPUT_DATA: [
                {
                    "fields": ["customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
                               "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                               "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
                               "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
                               "SampleWeight"],
                    "values":[["3638-WEABW","Female",0,"Yes","No",58,"Yes","Yes","DSL","No","Yes","No","Yes","No","No","Two year","Yes","Credit card (automatic)",59.9,3505.1,"No",2.768]]
                }
            ]
        }

        scores = self.wml_client.deployments.score(TestWMLClientWithSPSS.deployment_uid, scoring_payload)
        self.assertIsNotNone(scores)

    def test_10_delete_deployment(self):
        TestWMLClientWithSPSS.logger.info("Delete deployment")
        self.wml_client.deployments.delete(TestWMLClientWithSPSS.deployment_uid)

    def test_11_delete_model(self):
        TestWMLClientWithSPSS.logger.info("Delete model")
        self.wml_client.repository.delete(TestWMLClientWithSPSS.model_uid)


if __name__ == '__main__':
    unittest.main()
