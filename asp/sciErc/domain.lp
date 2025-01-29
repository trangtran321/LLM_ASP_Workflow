type_def("part-of","task","task").
type_def("part-of","generic","generic").
type_def("part-of","material","material").
type_def("part-of","otherscientificterm","otherscientificterm").
type_def("part-of","metric","metric").
type_def("part-of","method","method").
type_def("part-of","otherscientificterm","method").
type_def("part-of","generic","method").
type_def("part-of","method","generic").
type_def("part-of","task","method").

type_def("used-for","task","task").
type_def("used-for","method","task"). %%%%%
type_def("used-for","method","otherscientificterm").
type_def("used-for","otherscientificterm","otherscientificterm").
type_def("used-for","task","otherscientificterm").
type_def("used-for","generic","task"). %%%%%
type_def("used-for","method","method"). %%%%%
type_def("used-for","otherscientificterm","method").
type_def("used-for","material","task").
type_def("used-for","material","otherscientificterm").
type_def("used-for","material","method").
type_def("used-for","metric","method").


type_def("used-for","task","task").
type_def("used-for","task","generic").
type_def("used-for","task","otherscientificterm").
type_def("used-for","task","method").

type_def("used-for","generic","task").
type_def("used-for","generic","otherscientificterm").
type_def("used-for","generic","method").
type_def("used-for","generic","material").

type_def("used-for","method","task").
type_def("used-for","method","otherscientificterm").
type_def("used-for","method","generic").
type_def("used-for","method","material").
type_def("used-for","method","method").
type_def("used-for","method","metric").

type_def("used-for","otherscientificterm","otherscientificterm").
type_def("used-for","otherscientificterm","method").
type_def("used-for","otherscientificterm","task").
type_def("used-for","otherscientificterm","generic").

type_def("used-for","material","task").
type_def("used-for","material","method").
type_def("used-for","metric","task").
type_def("used-for","metric","method").
% type_def("used-for","material","generic").


type_def("hyponym-of","method","method").
type_def("hyponym-of","otherscientificterm","generic").
type_def("hyponym-of","otherscientificterm","otherscientificterm").
type_def("hyponym-of","method","otherscientificterm").
type_def("hyponym-of","method","task").
type_def("hyponym-of","material","task").
type_def("hyponym-of","metric","method").
type_def("hyponym-of","task","task").

type_def("conjunction","method","method").
type_def("conjunction","otherscientificterm","otherscientificterm").
type_def("conjunction","metric","metric").
type_def("conjunction","material","material").
type_def("conjunction","task","task").
type_def("conjunction","generic","generic").

type_def("feature-of","otherscientificterm","otherscientificterm").
type_def("feature-of","otherscientificterm","generic").
type_def("feature-of","otherscientificterm","material").
type_def("feature-of","otherscientificterm","metric").

type_def("evaluate-for","metric","method").
type_def("evaluate-for","material","method").
type_def("evaluate-for","method","generic").
type_def("evaluate-for","metric","generic").
type_def("evaluate-for","metric","otherscientificterm").
type_def("evaluate-for","generic","metric").
type_def("evaluate-for","material","generic").
type_def("evaluate-for","metric","task").

type_def("compare","otherscientificterm","otherscientificterm").
type_def("compare","method","method").
type_def("compare","generic","method").
type_def("compare","generic","material").
type_def("compare","generic","generic").
type_def("compare","method","generic").