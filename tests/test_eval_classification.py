import unittest
import pandas as pd
import eval_classification

# Type nosetests in commandline in project root to run these tests.
class TestEvalClassification(unittest.TestCase):

    def setUp(self):
        dict  = {"truth": [1,2,1,1],
                 "predicted": [1,1,1,2],
                 "confidence": [0.2,0.3,0.5,0.5]}
        self.df = pd.DataFrame(dict)

    # # Should see one passing test
    # def test_tests(self):
    #     self.assertEqual(1+1,2)

    # def test_example(self):
    #     self.assertEqual(1.0,1)
    #     self.assertAlmostEqual(1.000000001,1.000000002)
    #
    #
    def test_eval_classification(self):
        self.df.columns.contains("truth")
        self.df.columns.contains("predicted")
        self.df.columns.contains("confidence")
        rates=eval_classification.compare_thresholds(self.df, "truth", "predicted",
                        "confidence",0.4)
        #self.assertEqual(rates,(1,0,0))
        self.assertEqual(rates,
                         {"correct_classified_rate" : 0.25,
                            "misclassified_rate" : 0.25,
                            "not_confident_rate" : 0.5})
