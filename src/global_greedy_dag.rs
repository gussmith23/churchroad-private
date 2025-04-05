/// from https://github.com/egraphs-good/extraction-gym/blob/main/src/extract/global_greedy_dag.rs
use std::iter;

use egraph_serialize::Cost;
use log::debug;
use ordered_float::NotNan;
use rpds::HashTrieSet;

use super::*;

type TermId = usize;

pub const INFINITY: Cost = unsafe { NotNan::new_unchecked(f64::INFINITY) };

#[derive(Clone, PartialEq, Eq, Hash)]
struct Term {
    op: String,
    children: Vec<TermId>,
}

type Reachable = HashTrieSet<ClassId>;

struct TermInfo {
    node: NodeId,
    eclass: ClassId,
    node_cost: Cost,
    total_cost: Cost,
    // store the set of reachable terms from this term
    reachable: Reachable,
    size: usize,
}

/// A TermDag needs to store terms that share common
/// subterms using a hashmap.
/// However, it also critically needs to be able to answer
/// reachability queries in this dag `reachable`.
/// This prevents double-counting costs when
/// computing the cost of a term.
#[derive(Default)]
pub struct TermDag {
    nodes: Vec<Term>,
    info: Vec<TermInfo>,
    hash_cons: HashMap<Term, TermId>,
}

fn node_summary(node_id: &NodeId, egraph: &egraph_serialize::EGraph, depth: usize) -> String {
    let node = &egraph[node_id];
    let op_str = match node.op.as_str() {
        "Op0" | "Op1" | "Op2" | "Op3" => {
            format!("{} {}", node.op, egraph[&node.children[0]].op)
        }
        _ => node.op.clone(),
    };
    let class_str = if depth > 0 {
        format!(" ({})", class_summary(&node.eclass, egraph, depth - 1))
    } else {
        "".to_string()
    };

    format!("node {} with op {}{}", node_id, op_str, class_str)
}

fn class_summary(class_id: &ClassId, egraph: &egraph_serialize::EGraph, depth: usize) -> String {
    let class = &egraph[class_id];
    let nodes_str = if depth > 0 {
        format!(
            " (nodes: {})",
            class
                .nodes
                .iter()
                .map(|nid| node_summary(nid, egraph, depth - 1))
                .collect::<Vec<String>>()
                .join(", ")
        )
    } else {
        "".to_string()
    };
    format!("class {}{}", class_id, nodes_str)
}

impl TermDag {
    /// Makes a new term using a node and children terms
    /// Correctly computes total_cost with sharing
    /// If this term contains itself, returns None
    /// If this term costs more than target, returns None
    pub fn make(
        &mut self,
        node_id: NodeId,
        node: &Node,
        children: Vec<TermId>,
        target: Cost,
    ) -> Option<TermId> {
        let term = Term {
            op: node.op.clone(),
            children: children.clone(),
        };

        if let Some(id) = self.hash_cons.get(&term) {
            return Some(*id);
        }

        // NOTE: This is the only modification we made to make this work with
        // churchroad. Could find a different way to do this.
        let node_cost = node.cost;
        // let node_cost = match node.op.as_str() {
        // "Wire" => INFINITY,
        // "Shr" | "Shl" => {
        //     warn!("Shr and Shl probably shouldn't be extractable");
        //     10000.into()
        // }
        // "And" | "Add" | "Sub" | "Mul" | "Or" | "Xor" | "Eq" | "Ne" | "Not" | "ReduceOr"
        // | "ReduceAnd" | "ReduceXor" | "LogicNot" | "LogicAnd" | "LogicOr" | "Mux" => 10000.into(),
        //     _ => node.cost,
        // };

        if children.is_empty() {
            let next_id = self.nodes.len();
            self.nodes.push(term.clone());
            self.info.push(TermInfo {
                node: node_id,
                eclass: node.eclass.clone(),
                node_cost,
                total_cost: node_cost,
                reachable: iter::once(node.eclass.clone()).collect(),
                size: 1,
            });
            self.hash_cons.insert(term, next_id);
            Some(next_id)
        } else {
            // check if children contains this node, preventing cycles
            // This is sound because `reachable` is the set of reachable eclasses
            // from this term.
            for child in &children {
                if self.info[*child].reachable.contains(&node.eclass) {
                    return None;
                }
            }

            let biggest_child = (0..children.len())
                .max_by_key(|i| self.info[children[*i]].size)
                .unwrap();

            let mut cost = node_cost + self.total_cost(children[biggest_child]);
            let mut reachable = self.info[children[biggest_child]].reachable.clone();
            let next_id = self.nodes.len();

            for child in children.iter() {
                if cost > target {
                    return None;
                }
                let child_cost = self.get_cost(&mut reachable, *child);
                cost += child_cost;
            }

            if cost > target {
                return None;
            }

            reachable = reachable.insert(node.eclass.clone());

            self.info.push(TermInfo {
                node: node_id,
                node_cost,
                eclass: node.eclass.clone(),
                total_cost: cost,
                reachable,
                size: 1 + children.iter().map(|c| self.info[*c].size).sum::<usize>(),
            });
            self.nodes.push(term.clone());
            self.hash_cons.insert(term, next_id);
            Some(next_id)
        }
    }

    /// Return a new term, like this one but making use of shared terms.
    /// Also return the cost of the new nodes.
    fn get_cost(&self, shared: &mut Reachable, id: TermId) -> Cost {
        let eclass = self.info[id].eclass.clone();

        // This is the key to why this algorithm is faster than greedy_dag.
        // While doing the set union between reachable sets, we can stop early
        // if we find a shared term.
        // Since the term with `id` is shared, the reachable set of `id` will already
        // be in `shared`.
        if shared.contains(&eclass) {
            NotNan::<f64>::new(0.0).unwrap()
        } else {
            let mut cost = self.node_cost(id);
            for child in &self.nodes[id].children {
                let child_cost = self.get_cost(shared, *child);
                cost += child_cost;
            }
            *shared = shared.insert(eclass);
            cost
        }
    }

    pub fn node_cost(&self, id: TermId) -> Cost {
        self.info[id].node_cost
    }

    pub fn total_cost(&self, id: TermId) -> Cost {
        self.info[id].total_cost
    }
}

pub struct GlobalGreedyDagExtractor {
    pub structural_only: bool,
    /// Whether or not to fail on partial extraction, ie. if not all classes
    /// are extracted.
    pub fail_on_partial: bool,
    /// Predicate determining whether a node is extractable.
    pub extractable_predicate: fn(&egraph_serialize::EGraph, &NodeId) -> bool,
}
impl GlobalGreedyDagExtractor {
    /// - roots: apparently, roots is not necessary for running global greedy
    ///   extraction. Thus, this  can be empty. If roots is not empty, this
    ///   function will perform a check to see if complete expressions are
    ///   extracted for all roots.
    pub fn extract(
        &self,
        egraph: &egraph_serialize::EGraph,
        roots: &[ClassId],
    ) -> Result<IndexMap<ClassId, NodeId>, String> {
        let mut keep_going = true;

        let nodes = egraph.nodes.clone();
        let mut termdag = TermDag::default();
        let mut best_in_class: HashMap<ClassId, TermId> = HashMap::default();

        while keep_going {
            keep_going = false;

            'node_loop: for (node_id, node) in &nodes {
                if !(self.extractable_predicate)(egraph, node_id) {
                    continue 'node_loop;
                }

                let mut children: Vec<TermId> = vec![];
                // compute the cost set from the children
                for child in &node.children {
                    let child_cid = egraph.nid_to_cid(child);
                    if let Some(best) = best_in_class.get(child_cid) {
                        children.push(*best);
                    } else {
                        debug!(
                            "Skipping {}{} because child {} is missing",
                            if roots.contains(child_cid) {
                                "root "
                            } else {
                                ""
                            },
                            node_summary(node_id, egraph, 2),
                            class_summary(child_cid, egraph, 2)
                        );
                        continue 'node_loop;
                    }
                }

                let old_cost = best_in_class
                    .get(&node.eclass)
                    .map(|id| termdag.total_cost(*id))
                    .unwrap_or(INFINITY);

                if let Some(candidate) = termdag.make(node_id.clone(), node, children, old_cost) {
                    let cadidate_cost = termdag.total_cost(candidate);

                    if cadidate_cost < old_cost {
                        best_in_class.insert(node.eclass.clone(), candidate);
                        keep_going = true;
                        debug!(
                            "Node {} (class {}) cost {} -> {}",
                            node_id, node.eclass, old_cost, cadidate_cost
                        );
                    }
                }
            }
        }

        let mut result = IndexMap::default();
        for (class, term) in best_in_class {
            result.insert(class, termdag.info[term].node.clone());
        }

        let missing = egraph
            .classes()
            .iter()
            .filter(|&(cid, _)| !result.contains_key(cid))
            .collect::<Vec<_>>();

        fn display_node(node: &Node, egraph: &egraph_serialize::EGraph) -> String {
            match node.op.as_str() {
                "Op0" | "Op1" | "Op2" | "Op3" => {
                    format!("{} {}", node.op, egraph[&node.children[0]].op)
                }
                _ => node.op.clone(),
            }
        }

        fn display_eclass(cid: &ClassId, egraph: &egraph_serialize::EGraph) -> String {
            egraph[cid]
                .nodes
                .iter()
                .map(|nid| display_node(&egraph[nid], egraph))
                .collect::<Vec<_>>()
                .join(", ")
        }

        if !Vec::from(roots).is_empty() {
            let mut seen_classes: HashSet<_> = HashSet::new();
            let mut queue: Vec<_> = roots.iter().collect::<Vec<_>>();
            let mut needed_classes = HashSet::new();

            while let Some(cid) = queue.pop() {
                
                if seen_classes.contains(cid) {
                    continue;
                }
                seen_classes.insert(cid);

                if let Some(node) = &result.get(cid) {
                    for child in &egraph[*node].children {
                        let child_cid = egraph.nid_to_cid(child);
                        if !seen_classes.contains(child_cid) {
                            queue.push(child_cid);
                        }
                    }
                } else {
                    needed_classes.insert(cid);
                }
            }

            if !needed_classes.is_empty() {
                return Err(
                    "Not all roots were able to be extracted. Missing classes:\n".to_string()
                        + &needed_classes
                            .iter()
                            .map(|cid| format!("{}: {}", cid, display_eclass(cid, egraph)))
                            .collect::<Vec<_>>()
                            .join("\n"),
                );
            }
        }

        if self.fail_on_partial && !missing.is_empty() {
            Err(
                "Not all classes were able to be extracted. Missing classes:\n".to_string()
                    + &missing
                        .iter()
                        .map(|(cid, _)| format!("{}: {}", cid, display_eclass(cid, egraph)))
                        .collect::<Vec<_>>()
                        .join("\n"),
            )
        } else {
            Ok(result)
        }
    }
}
