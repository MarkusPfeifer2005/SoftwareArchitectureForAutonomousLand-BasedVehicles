#!/usr/bin/env python
import unittest
import os
import shutil
import numpy as np

from init import Config
from benchmark.nearest_neighbour import Cifar10Dataset, ManhattanModel, train, evaluate
from mlib.scratch import SVMLossVectorized, WeightMultiplication, BiasAddition, SigmoidLayer,\
    LinearLayer, MathematicalFunc, Model, StochasticGradientDecent, Layer, MSE
from benchmark.linear_classification import ExperimentalModel, LinearClassifier


def gradient_check(m_func: MathematicalFunc, x: np.ndarray, d: float = 1e-4) -> list:
    m_func.forward(x)  # Forward pass just in case so all params are in place.
    gradients = []
    for idx, param in enumerate(m_func.parameters):  # gradients for all internal parameters
        grad_numerical = np.zeros_like(m_func.parameters[idx])
        with np.nditer(grad_numerical, flags=["multi_index"], op_flags=["readwrite"]) as it:
            for element in it:
                m_func.parameters[idx][it.multi_index] += d
                fxd = m_func.forward(x)
                m_func.parameters[idx][it.multi_index] -= d  # the '=' operator would destroy the reference!
                fx = m_func.forward(x)
                dw = (fxd - fx) / d
                element[...] = np.sum(dw)
        gradients.append(grad_numerical)
    return gradients


def gradient_check_model(model: Model, x: np.ndarray, d: float = 1e-4) -> list:
    model.forward(x)  # Forward pass just in case so all params are in place.
    gradients = []
    for layer in reversed(model.layers):
        for idx, param in enumerate(layer.parameters):
            grad_numerical = np.zeros_like(layer.parameters[idx])
            with np.nditer(grad_numerical, flags=["multi_index"], op_flags=["readwrite"]) as it:
                for element in it:
                    layer.parameters[idx][it.multi_index] += d
                    fxd = model.forward(x)
                    layer.parameters[idx][it.multi_index] -= d  # the '=' operator would destroy the reference!
                    fx = model.forward(x)
                    dw = (fxd - fx) / d
                    element[...] = np.sum(dw)
                gradients.append(grad_numerical)
    return gradients


class TestMyDataset(unittest.TestCase):
    def setUp(self):
        config = Config(root="config.json")
        self.dataset = Cifar10Dataset(config["cifar"])

    def test_num_classes(self):
        self.assertEqual(self.dataset.num_classes, 10)

    def test___iter__(self):
        for batch in self.dataset:
            self.assertIsInstance(batch[b"batch_label"], bytes)
            self.assertIsInstance(batch[b"labels"], list)
            self.assertIsInstance(batch[b"data"], np.ndarray)
            self.assertIsInstance(batch[b"filenames"], list)


class TestTrain(unittest.TestCase):
    def test_train(self):
        config = Config(root="config.json")
        train_dataset = Cifar10Dataset(root=config["cifar"])
        model = ManhattanModel()
        train(model=model, dataset=train_dataset)

        # check data
        d_samples = 0
        l_samples = 0
        for batch in train_dataset:
            d_samples += len(batch[b"data"])
            l_samples += len(batch[b"labels"])
        self.assertEqual(len(model.data), d_samples)
        self.assertEqual(len(model.labels), l_samples)


class TestManhattanModel(unittest.TestCase):
    def test___call__(self):
        config = Config(root="config.json")
        model: ManhattanModel = ManhattanModel()
        dataset: Cifar10Dataset = Cifar10Dataset(batches=slice(0, 1), root=config["cifar"])
        train(model, dataset)

        for batch in dataset:
            for img, lbl in zip(batch[b"data"][0:20], batch[b"labels"][0:20]):
                target = dataset.labels[lbl].decode()
                prediction = dataset.labels[model(img)].decode()

                self.assertEqual(prediction, target)


class TestEvaluate(unittest.TestCase):
    def test_evaluate(self):
        config = Config(root="config.json")
        model = ManhattanModel()
        dataset = Cifar10Dataset(batches=slice(0, 1), root=config["cifar"])
        train(model, dataset)
        self.assertEqual(evaluate(model, dataset, images=slice(0, 60)), 100.00)


class TestSVMLossVectorized(unittest.TestCase):
    def setUp(self):
        self.criterion = SVMLossVectorized()

    def test_forward(self):
        scores = np.array([
            [3.2, 5.1, -1.7],  # 5.1-3.2+1 = 2.9
            [1.3, 4.9, 2.0],   # 0 = 0
            [2.2, 2.5, -3.1]   # 2.2-(-3.1)+1 + 2.5-(-3.1)+1 = 12.9
        ])
        targets = [0, 1, 2]
        loss = self.criterion.forward(x=scores, y=targets)
        self.assertEqual(round(15.8/scores.size, 5), np.round(loss, 5).tolist())

    def test_backward(self):
        scores = np.array([
            [3.2, 5.1, -1.7],
            [1.3, 4.9, 2.0],
            [2.2, 2.5, -3.1]
        ])
        targets = [0, 1, 2]
        self.criterion.forward(x=scores, y=targets)  # calculates losses
        grads = self.criterion.backward()

        self.assertEqual(scores.shape, grads.shape)
        self.assertEqual([[-0.1111111111111111, 0.1111111111111111, 0.0],
                          [0.0, -0.0, 0.0],
                          [0.1111111111111111, 0.1111111111111111, -0.2222222222222222]], grads.tolist())


class TestWeightMultiplication(unittest.TestCase):
    def test_forward(self):
        x = np.array([[8, 6],
                      [1, 5],
                      [2, 6]])
        w = np.array([[5, 9],
                      [7, 2]])
        f = WeightMultiplication(w)
        s = f.forward(x)
        self.assertEqual([[82, 84],
                          [40, 19],
                          [52, 30]], s.tolist())

    def test_backward(self):
        w = np.array([[.2],
                      [.4]])
        x = np.array([[.1, .5],
                      [-.3, .8]])
        f = WeightMultiplication(w)
        f.forward(x)  # calculates scores
        dx, dw = f.backward(np.array([[.44], [.52]]))
        dw = dw[0]

        self.assertEqual([[-0.11199999999999999],
                          [0.636]], dw.tolist())
        self.assertEqual([[0.08800000000000001, 0.17600000000000002],
                          [0.10400000000000001, 0.20800000000000002]], dx.tolist())
        self.assertEqual(w.shape, dw.shape)
        self.assertEqual(x.shape, dx.shape)

        x = np.array([[6, 5, 2, 8],                     # [[x11, x12, x13, x14],
                      [4, 7, 8, 2]]).astype("float64")  #  [x21, x22, x23, x24]]
        w = np.array([[7, 8, 2],                        # [[w11, w12, w13],
                      [1, 0, 5],                        #  [w21, w22, w23],
                      [1, 3, 3],                        #  [w31, w32, w33],
                      [9, 8, 1]]).astype("float64")     #  [w41, w42, w43]]
        f = WeightMultiplication(w)
        s = f.forward(x)  # [[(x11*w11+x12*w21+x13*w31+x14*w41), (x11*w12+x12*w22+x13*w32+x14*w42), (x11*w13+x12*w23+x13*w33+x14*w43)],
        #                    [(x21*w11+x22*w21+x23*w31+x24*w41), (x21*w12+x22*w22+x13*w32+x14*w42), (x21*w13+x22*w23+x23*w33+x24*w43)]]

        ds = np.ones_like(s)
        dx_anl, dw_anl = f.backward(prev_grad=ds)
        dw_anl = dw_anl[0]
        dw_num = gradient_check(f, x)[0]

        self.assertEqual(x.shape, dx_anl.shape)
        self.assertEqual(w.shape, dw_anl.shape)
        self.assertEqual(w.shape, dw_num.shape)
        self.assertEqual(np.round(dw_num, 3).tolist(), np.round(dw_anl, 3).tolist())


class TestBiasAddition(unittest.TestCase):
    """This thing is vectorized!!"""
    def test_forward(self):
        x = np.array([[2, 3],
                      [4, 9]])
        b = np.array([[1, 7],
                      [2, 4]])
        addition = BiasAddition(b)
        _sum = addition.forward(x)
        self.assertEqual([[3, 10],
                          [6, 13]], _sum.tolist())

    def test_backward(self):
        x = np.random.randint(-10, 10, size=(10, 3)).astype("float64")
        b = np.random.randint(-10, 10, size=(3, )).astype("float64")
        f = BiasAddition(b)
        s = f.forward(x)

        ds = np.ones_like(s)
        dx_anl, db_anl = f.backward(prev_grad=ds)
        db_anl = db_anl[0]
        db_num = gradient_check(f, x)[0]

        self.assertEqual(x.shape, dx_anl.shape)
        self.assertEqual(b.shape, db_anl.shape)
        self.assertEqual(b.shape, db_num.shape)
        self.assertEqual(np.round(db_num, 3).tolist(), np.round(db_anl, 3).tolist())


class TestLayer(unittest.TestCase):
    def setUp(self):
        self.layers = [LinearLayer(),
                       SigmoidLayer()]

    def test_backward(self):
        data = np.random.randint(0, 255, (2, 3072)).astype("float64")
        data /= 255  # Normalize data to fit between 0 and 1.

        for layer in self.layers:
            numerical_grads: list = gradient_check(layer, data, d=1e-6)
            analytical_grads = layer.backward(np.ones_like(layer.forward(data)))[1]

            # compare gradients
            for num_grad, anl_grad, param in zip(numerical_grads, analytical_grads, layer.parameters):
                self.assertEqual(param.shape, num_grad.shape)
                self.assertEqual(param.shape, anl_grad.shape)
                self.assertEqual(np.round(num_grad, 2).tolist(), np.round(anl_grad, 2).tolist())


class TestModel(unittest.TestCase):
    test_dir_name = "test-files"

    def setUp(self):
        os.mkdir(self.test_dir_name)
        self.criterion = SVMLossVectorized()
        self.models = [
            LinearClassifier(),
            ExperimentalModel()
        ]

    def tearDown(self):
        try:
            shutil.rmtree(self.test_dir_name)
        except FileNotFoundError:
            pass

    def test_forward(self):
        data = np.random.randint(0, 255, size=(9, 3072))
        for model in self.models:
            scores = model.forward(data)
            self.assertIsInstance(scores, np.ndarray)
            self.assertEqual((9, 10), scores.shape)

    def test_backward(self):
        data = np.random.randint(0, 255, size=(2, 3072))
        for model in self.models:
            scores = model.forward(data)
            loss_grad = np.ones_like(scores)

            numerical_grads = gradient_check_model(model, data, d=1e-4)
            analytical_grads = []
            grad = loss_grad
            for layer in reversed(model.layers):
                grad, parameter_grads = layer.backward(grad)
                analytical_grads += parameter_grads

            # compare gradients
            for num_grad, anl_grad in zip(numerical_grads, analytical_grads):
                self.assertEqual(np.round(num_grad, 2).tolist(), np.round(anl_grad, 2).tolist())

    def test_save(self):
        for model in self.models:
            model.save(path=self.test_dir_name, epoch=0)
            self.assertTrue(os.path.isfile(os.path.join(self.test_dir_name, f"{model.file_prefix}{0}")))

    def test_load(self):
        config = Config(root="config.json")
        evaluation_set: Cifar10Dataset = Cifar10Dataset(batches=slice(4, 5), root=config["cifar"])
        for model in self.models:
            accuracy1 = evaluate(model, evaluation_set)
            model.save(path=self.test_dir_name, epoch=0)
            model.load(self.test_dir_name)
            accuracy2 = evaluate(model, evaluation_set)
            self.assertEqual(accuracy1, accuracy2)


class DummyLayer(Layer):
    def __init__(self, num_pixels: int = 3072, num_classes: int = 10):
        super().__init__()
        self.parameters = [np.ones(shape=(num_pixels, num_classes)),
                           np.ones(shape=(num_classes,))]
        self.operations = [WeightMultiplication(weight=self.parameters[0]), BiasAddition(bias=self.parameters[1])]

    def forward(self, x) -> np.ndarray:
        self.x = x
        for operation in self.operations:
            x = operation.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray, list]:
        return np.ones_like(self.x), [np.ones_like(self.parameters[0]), np.ones_like(self.parameters[1])]


class TestDummyLayer(unittest.TestCase):
    def setUp(self):
        self.layer1 = DummyLayer(num_pixels=4, num_classes=3)

    def test_forward(self):
        x = np.ones(shape=(5, 4))
        s = self.layer1.forward(x)
        self.assertEqual((5, 3), s.shape)
        self.assertEqual(np.full(shape=(5, 3), fill_value=5.).tolist(), s.tolist())

    def test_backward(self):
        x = np.ones(shape=(5, 4))
        s = self.layer1.forward(x)

        ds = np.ones_like(s)
        dx = self.layer1.backward(ds)[0]
        dw, db = self.layer1.backward(ds)[1]

        self.assertEqual(x.shape, dx.shape)
        self.assertEqual(self.layer1.parameters[0].shape, dw.shape)
        self.assertEqual(self.layer1.parameters[1].shape, db.shape)

        self.assertEqual(np.ones_like(x).tolist(), dx.tolist())
        self.assertEqual(np.ones_like(self.layer1.parameters[0]).tolist(), dw.tolist())
        self.assertEqual(np.ones_like(self.layer1.parameters[1]).tolist(), db.tolist())


class DummyModel(Model):
    def __init__(self):
        super().__init__()
        self.layers = [DummyLayer(num_pixels=3072, num_classes=10),
                       DummyLayer(num_pixels=10, num_classes=10)]


class TestDummyModel(unittest.TestCase):
    def setUp(self):
        self.model = DummyModel()

    def test_forward(self):
        x = np.random.randint(0, 255, size=(9, 3072))
        s = self.model.forward(x)
        self.assertEqual((9, 10), s.shape)

    def test_backward(self):
        x = np.random.randint(0, 255, size=(9, 3072))
        s = self.model.forward(x)

        ds = np.ones_like(s)
        ds, dwb = self.model.layers[1].backward(ds)
        dw1, db1 = dwb[0], dwb[1]
        _, dwb = self.model.layers[0].backward(ds)
        dw0, db0 = dwb[0], dwb[1]

        self.assertEqual(np.ones_like(self.model.layers[1].parameters[0]).tolist(), dw1.tolist())
        self.assertEqual(np.ones_like(self.model.layers[1].parameters[1]).tolist(), db1.tolist())
        self.assertEqual(np.ones_like(self.model.layers[0].parameters[0]).tolist(), dw0.tolist())
        self.assertEqual(np.ones_like(self.model.layers[0].parameters[1]).tolist(), db0.tolist())


class TestOptimizer(unittest.TestCase):
    def setUp(self):
        self.model = DummyModel()
        self.lr = 0.001
        self.optim = StochasticGradientDecent(model_layers=self.model.layers, lr=self.lr)

    def test_step(self):
        data = np.random.randint(0, 255, size=(9, 3072))
        scores = self.model.forward(data)
        d_scores = np.ones_like(scores)
        self.optim.step(grad=d_scores)

        self.assertEqual(np.full(shape=(3072, 10), fill_value=1-self.lr).tolist(),
                         self.model.layers[0].parameters[0].tolist())
        self.assertEqual(np.full(shape=(10, ), fill_value=1-self.lr).tolist(),
                         self.model.layers[0].parameters[1].tolist())
        self.assertEqual(np.full(shape=(10, 10), fill_value=1-self.lr).tolist(),
                         self.model.layers[1].parameters[0].tolist())
        self.assertEqual(np.full(shape=(10, ), fill_value=1-self.lr).tolist(),
                         self.model.layers[1].parameters[1].tolist())


class TestMSE(unittest.TestCase):
    def test_forward(self):
        x = np.array([[1.], [2.]])
        y = np.array([[1.], [4.]])

        l = MSE()
        loss = l(x, y)
        self.assertEqual(2., loss)

    def test_backward(self):
        x = np.array([[1.], [2.]])
        y = np.array([[1.], [4.]])

        l = MSE()
        l(x, y)  # calculates loss
        dldx = l.backward()

        self.assertEqual([[0.], [-2.]], dldx.tolist())


if __name__ == "__main__":
    unittest.main()
