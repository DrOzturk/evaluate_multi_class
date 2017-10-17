import unittest
import pandas as pd
import eval_classification

# Type nosetests in commandline in project root to run these tests.
class TestEvalClassification(unittest.TestCase):

    def setUp(self):
        dict  = {"truth": [1,2,2,1],
                 "predicted": [1,1,2,2],
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
    def test_eval_with_threshold(self):
        self.df.columns.contains("truth")
        self.df.columns.contains("predicted")
        self.df.columns.contains("confidence")

        # If confidence cutoff is .5
        rates=eval_classification.eval_with_threshold(self.df, "truth", "predicted",
                        "confidence", 0.4)
        #self.assertEqual(rates,(1,0,0))
        self.assertEqual(rates,
                         {"correct_classified_rate" : 0.25,
                            "misclassified_rate" : 0.25,
                            "not_confident_rate" : 0.5})

        #as we decrease required confidence, we expect decrease in non_confident
        # and possible increase of misclassified or correct classified
        rates=eval_classification.eval_with_threshold(self.df, "truth", "predicted",
                        "confidence", 0.3)
        #self.assertEqual(rates,(1,0,0))
        self.assertEqual(rates,
                         {"correct_classified_rate" : 0.25,
                            "misclassified_rate" : 0.5,
                            "not_confident_rate" : 0.25})

        #further decrease required confidence, further decrease in non_confident
        # and possible increase of misclassified or correct classified
        rates=eval_classification.eval_with_threshold(self.df, "truth", "predicted",
                        "confidence", 0.2)
        #self.assertEqual(rates,(1,0,0))
        self.assertEqual(rates,
                         {"correct_classified_rate" : 0.5,
                            "misclassified_rate" : 0.5,
                            "not_confident_rate" : 0})

    def test_run_thresholds(self):
        thresholds = [ 0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9]
        rates_on_range = eval_classification.run_thresholds(self.df, "truth", "predicted",
                        "confidence", thresholds)
        self.assertEqual(len(thresholds), rates_on_range.shape[0])
        self.assertEqual(4,rates_on_range.shape[1])
        # third one is threshold 0.3
        pd.testing.assert_series_equal(rates_on_range.loc[2], pd.Series({"correct_classified_rate" : 0.25,
                            "misclassified_rate" : 0.5,
                            "not_confident_rate" : 0.25,
                            "threshold" : 0.3}),check_names=False)

