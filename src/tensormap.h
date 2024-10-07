/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw © 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/
#pragma once
#include "tensor.h"
#include "op_base.h"

#include <unordered_map>
#include <memory>
#include <tuple>
#include <variant>
#include <string>
#include <string_view>

class TensorMap {
public:
	using Map = std::unordered_map< std::string, TensorMap >;
	using variant_t = std::variant< Map, Tensor >;
private:
	/* VARIANT TYPE */
	variant_t v;

	std::vector< TensorMap* > node_refs;
	std::vector< TensorMap* >& get_node_refs() { return root_ptr->node_refs; }
	const std::vector< TensorMap* >& get_node_refs() const { return root_ptr->node_refs; }
	index_t get_next_id() const { return get_node_refs().size(); }

	TensorMap* root_ptr = nullptr;
	TensorMap* parent_ptr = nullptr;
	const index_t id_ = 0;
	std::string_view name_;

	/* Private constructor used to emplace nodes */
	TensorMap(TensorMap* parent_ptr) :
		root_ptr{ parent_ptr->root_ptr }, parent_ptr{ parent_ptr }, id_{ get_next_id() }
	{}
public:
	/* Root constructor */
	TensorMap() {
		this->root_ptr = this;
		this->parent_ptr = this;
		node_refs.push_back(this);
	}

	/* Testers */
	bool is_root() const { return root_ptr == this; }
	bool is_map() const { return std::holds_alternative< Map >(v); }
	bool is_tensor() const { return std::holds_alternative< Tensor >(v); }
	bool has(const std::string& s) const {
		return is_map() && map().find(s) != map().end();
	}
	bool has(const index_t& id) const {
		const auto& node_refs_vec = get_node_refs();
		if (id < 0 || id >= index_t(node_refs_vec.size())) return false;
		return node_refs_vec[id] != nullptr;
	}

	/* Getters */
	TensorMap& root() const { return *this->root_ptr; }
	TensorMap& parent() const { return *this->parent_ptr; }
	index_t id() const { return is_root() ? 0 : this->id_; }
	Map& map() { return std::get< Map >(v); }
	const Map& map() const { return std::get< Map >(v); }

	Tensor& tensor() { return std::get< Tensor >(v); }
	const Tensor& tensor() const { return std::get< Tensor >(v); }

	const std::string_view name() const { return name_; }

	/* Setters. FIXME a change should never erase tensor access so if the node is a map,
	the map must be empty */
	void operator=(const double& x) {
		assert(((is_tensor() == true) || (map().size() == 0)) && "Cannot erase an empty map.");
		v = Tensor{ x };
	}
	void operator=(const std::vector< double >&& x) { v = Tensor{ std::forward< const std::vector< double > >(x) }; }
	void operator=(const std::vector< double >& x) { v = Tensor{ x }; }
	void operator=(const TensorMap& x) { v = x.v; }
	void operator=(const Tensor& x) { v = x; }
	void operator=(const std::string& s) { this->operator[](s); }

	/* Map getter/setter */
	TensorMap& operator[](const std::string& s) {

		auto it = this->map().find(s);

		if (it == this->map().end()) {
			/* Create node in-place */
			auto ret = this->map().emplace(s, TensorMap(this));
			/* Keep pointer to name in the created node */
			ret.first->second.name_ = { ret.first->first };
			/* Keep pointer to the created node in the random access vector */
			get_node_refs().push_back(&ret.first->second);
			return ret.first->second;
		}
		else {
			return it->second;
		}
	}
	/* Random access */
	TensorMap& operator[](const index_t id) {
		assert(this->has(id));
		return *get_node_refs()[id];
	}

	const TensorMap& operator[](const index_t id) const {
		assert(this->has(id));
		return *get_node_refs()[id];
	}
};