/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw © 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/
#pragma once
#include "op_base.h"

/****************************************************************************
*** IVERSON BRACKETS, LINEAR SCALE
*****************************************************************************/

/* y = [x > 0] */
struct GreaterThanZeroScalar : OpIn::Array<1>, OpConst::None, OpOut::Scalar, PartialAlwaysZero, HessianAlwaysZero {

	GreaterThanZeroScalar(const OpIn::Array<1> in, const OpConst::None constant, const OpOut::Scalar out) :
		OpIn::Array<1>{ in }, OpOut::Scalar{ out } {}

	void evaluate(std::vector< double >& v) const {
		double& y = v[out[0]];
		double& x = v[in[0]];
		y = x > 0.0 ? 1.0 : 0.0;
	}
	struct LocalDiff {
		constexpr double partial(const index_t i, const index_t j) const {
			return 0.;
		}
		constexpr double partial(const index_t i, const index_t j, const index_t k) const {
			return 0.;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{};
	}
};
/* y = [x > 0] */
struct GreaterThanZeroVector : OpIn::Range, OpConst::None, OpOut::Range,
	PartialAlwaysZero, HessianAlwaysZero, OpOutSize::Range_None_RangeSize
{
	GreaterThanZeroVector(const OpIn::Range in, const OpConst::None constant, const OpOut::Range out) :
		OpIn::Range{ in }, OpOut::Range{ out }
	{
		assert(this->in.size() == this->out.size());
	}
	void evaluate(std::vector< double >& v) const {
		assert(v.size() >= in.size() + out.size());
		assert(in.size() == out.size());
		for (index_t i = 0; i < index_t(in.size()); i++) {
			v[out[i]] = v[in[i]] > 0.0 ? 1.0 : 0.0;
		}
	}
	struct LocalDiff {
		constexpr double partial(const index_t i, const index_t j) const {
			return 0.;
		}
		constexpr double partial(const index_t i, const index_t j, const index_t k) const {
			return 0.;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{};
	}
};
/* y = [x >= 0] */
struct GreaterThanOrEqualZeroScalar : OpIn::Array<1>, OpConst::None, OpOut::Scalar, PartialAlwaysZero, HessianAlwaysZero {

	GreaterThanOrEqualZeroScalar(const OpIn::Array<1> in, const OpConst::None constant, const OpOut::Scalar out) :
		OpIn::Array<1>{ in }, OpOut::Scalar{ out } {}

	void evaluate(std::vector< double >& v) const {
		double& y = v[out[0]];
		double& x = v[in[0]];
		y = x >= 0.0 ? 1.0 : 0.0;
	}
	struct LocalDiff {
		constexpr double partial(const index_t i, const index_t j) const {
			return 0.;
		}
		constexpr double partial(const index_t i, const index_t j, const index_t k) const {
			return 0.;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{};
	}
};
/* y = [x >= 0] */
struct GreaterThanOrEqualZeroVector : OpIn::Range, OpConst::None, OpOut::Range,
	PartialAlwaysZero, HessianAlwaysZero, OpOutSize::Range_None_RangeSize
{
	GreaterThanOrEqualZeroVector(const OpIn::Range in, const OpConst::None constant, const OpOut::Range out) :
		OpIn::Range{ in }, OpOut::Range{ out }
	{
		assert(this->in.size() == this->out.size());
	}
	void evaluate(std::vector< double >& v) const {
		assert(v.size() >= in.size() + out.size());
		assert(in.size() == out.size());
		for (index_t i = 0; i < index_t(in.size()); i++) {
			v[out[i]] = v[in[i]] >= 0.0 ? 1.0 : 0.0;
		}
	}
	struct LocalDiff {
		constexpr double partial(const index_t i, const index_t j) const {
			return 0.;
		}
		constexpr double partial(const index_t i, const index_t j, const index_t k) const {
			return 0.;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{};
	}
};

/****************************************************************************
*** IVERSON BRACKETS, LOG SCALE
*****************************************************************************/

/* y = log[x > 0] */
struct LogGreaterThanZeroScalar : OpIn::Array<1>, OpConst::None, OpOut::Scalar, PartialAlwaysZero, HessianAlwaysZero {

	LogGreaterThanZeroScalar(const OpIn::Array<1> in, const OpConst::None constant, const OpOut::Scalar out) :
		OpIn::Array<1>{ in }, OpOut::Scalar{ out } {}

	void evaluate(std::vector< double >& v) const {
		double& y = v[out[0]];
		double& x = v[in[0]];
		y = x > 0.0 ? 0.0 : -std::numeric_limits<double>::infinity();
	}
	struct LocalDiff {
		constexpr double partial(const index_t i, const index_t j) const {
			return 0.;
		}
		constexpr double partial(const index_t i, const index_t j, const index_t k) const {
			return 0.;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{};
	}
};
/* y = log[x > 0] */
struct LogGreaterThanZeroVector : OpIn::Range, OpConst::None, OpOut::Range,
	PartialAlwaysZero, HessianAlwaysZero, OpOutSize::Range_None_RangeSize
{
	LogGreaterThanZeroVector(const OpIn::Range in, const OpConst::None constant, const OpOut::Range out) :
		OpIn::Range{ in }, OpOut::Range{ out }
	{
		assert(this->in.size() == this->out.size());
	}
	void evaluate(std::vector< double >& v) const {
		assert(v.size() >= in.size() + out.size());
		assert(in.size() == out.size());
		for (index_t i = 0; i < index_t(in.size()); i++) {
			v[out[i]] = v[in[i]] > 0.0 ? 0.0 : -std::numeric_limits<double>::infinity();
		}
	}
	struct LocalDiff {
		constexpr double partial(const index_t i, const index_t j) const {
			return 0.;
		}
		constexpr double partial(const index_t i, const index_t j, const index_t k) const {
			return 0.;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{};
	}
};
/* y = log[x >= 0] */
struct LogGreaterThanOrEqualZeroScalar : OpIn::Array<1>, OpConst::None, OpOut::Scalar, PartialAlwaysZero, HessianAlwaysZero {

	LogGreaterThanOrEqualZeroScalar(const OpIn::Array<1> in, const OpConst::None constant, const OpOut::Scalar out) :
		OpIn::Array<1>{ in }, OpOut::Scalar{ out } {}

	void evaluate(std::vector< double >& v) const {
		double& y = v[out[0]];
		double& x = v[in[0]];
		y = x >= 0.0 ? 0.0 : -std::numeric_limits<double>::infinity();
	}
	struct LocalDiff {
		constexpr double partial(const index_t i, const index_t j) const {
			return 0.;
		}
		constexpr double partial(const index_t i, const index_t j, const index_t k) const {
			return 0.;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{};
	}
};
/* y = log[x >= 0] */
struct LogGreaterThanOrEqualZeroVector : OpIn::Range, OpConst::None, OpOut::Range,
	PartialAlwaysZero, HessianAlwaysZero, OpOutSize::Range_None_RangeSize
{
	LogGreaterThanOrEqualZeroVector(const OpIn::Range in, const OpConst::None constant, const OpOut::Range out) :
		OpIn::Range{ in }, OpOut::Range{ out }
	{
		assert(this->in.size() == this->out.size());
	}
	void evaluate(std::vector< double >& v) const {
		assert(v.size() >= in.size() + out.size());
		assert(in.size() == out.size());
		for (index_t i = 0; i < index_t(in.size()); i++) {
			v[out[i]] = v[in[i]] >= 0.0 ? 0.0 : -std::numeric_limits<double>::infinity();
		}
	}
	struct LocalDiff {
		constexpr double partial(const index_t i, const index_t j) const {
			return 0.;
		}
		constexpr double partial(const index_t i, const index_t j, const index_t k) const {
			return 0.;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{};
	}
};

/****************************************************************************
*** OPERATOR DECLARATION
*****************************************************************************/

using op_iverson_types = std::tuple<
	GreaterThanZeroScalar, GreaterThanZeroVector,
	GreaterThanOrEqualZeroScalar, GreaterThanOrEqualZeroVector,
	LogGreaterThanZeroScalar, LogGreaterThanZeroVector,
	LogGreaterThanOrEqualZeroScalar, LogGreaterThanOrEqualZeroVector
>;