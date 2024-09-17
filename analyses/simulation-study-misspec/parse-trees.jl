using gcdyn, JLD2
import JSON

EVENT_MAPPING = Dict(
	"birth" => :birth,
	"mutation" => :type_change,
	"sampling" => :sampled_survival,
	"root" => :root
)

function main()
	trees = map(TreeNode, JSON.parsefile("trees.json"))
	save_object("trees.jld2", trees)
end

function gcdyn.TreeNode(json_tree::Dict)
	root = TreeNode(
		EVENT_MAPPING[json_tree["event"]],
		json_tree["time"],
		json_tree["affinity"]
	)

	for child in json_tree["children"]
		attach!(root, TreeNode(child))
	end

	return root
end

main()