"use client";

import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  BackgroundVariant,
  type Node,
  type NodeChange,
  type EdgeChange,
  applyNodeChanges,
  applyEdgeChanges,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useStore } from "@/lib/store";
import ModelNode from "./nodes/ModelNode";
import SheafNode from "./nodes/SheafNode";
import MechanicNode from "./nodes/MechanicNode";
import StructureNode from "./nodes/StructureNode";
import DynamicNode from "./nodes/DynamicNode";
import { useCallback } from "react";

const NODE_TYPES = {
  model: ModelNode,
  sheaf: SheafNode,
  mechanic: MechanicNode,
  structure: StructureNode,
  dynamic: DynamicNode,
};

export default function StudioCanvas() {
  const { nodes, edges, setNodes, setEdges, setSelectedNode, activeArch, archLoading } =
    useStore();

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => setNodes(applyNodeChanges(changes, nodes)),
    [nodes, setNodes]
  );

  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => setEdges(applyEdgeChanges(changes, edges)),
    [edges, setEdges]
  );

  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => setSelectedNode(node),
    [setSelectedNode]
  );

  const onPaneClick = useCallback(() => setSelectedNode(null), [setSelectedNode]);

  if (!activeArch) {
    return (
      <div
        style={{
          flex: 1,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          flexDirection: "column",
          gap: 12,
          color: "var(--text-dim)",
          userSelect: "none",
        }}
      >
        <div style={{ fontSize: 32 }}>⬡</div>
        <div style={{ fontSize: 13 }}>Select an architecture from the left panel</div>
      </div>
    );
  }

  if (archLoading) {
    return (
      <div
        style={{
          flex: 1,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "var(--text-muted)",
        }}
      >
        loading…
      </div>
    );
  }

  return (
    <div style={{ flex: 1, position: "relative" }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={NODE_TYPES}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}
        fitView
        fitViewOptions={{ padding: 0.3 }}
        minZoom={0.2}
        maxZoom={2}
        proOptions={{ hideAttribution: true }}
        colorMode="dark"
      >
        <Background
          variant={BackgroundVariant.Dots}
          gap={24}
          size={1}
          color="var(--border)"
        />
        <Controls showInteractive={false} />
        <MiniMap
          nodeColor={(n) => {
            const t = n.type;
            if (t === "model") return "var(--blue)";
            if (t === "sheaf") return "var(--purple)";
            if (t === "structure") return "var(--orange)";
            if (t === "dynamic") return "var(--red)";
            return "var(--accent)";
          }}
          maskColor="rgba(0,0,0,0.5)"
          style={{ background: "var(--bg-panel)" }}
        />
      </ReactFlow>
    </div>
  );
}
