import React, { useMemo } from 'react';
import { ReactFlow, Background, Controls, Edge as ReactFlowEdge, MarkerType, EdgeProps, getStraightPath, BaseEdge, EdgeLabelRenderer, Handle, Position } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { EdgeData, NodeData } from '../types';

interface NetworkGraphProps {
  nodesData?: NodeData[];
  edgesData: EdgeData[];
  congestionData: Record<string, number>;
  thresholds: Record<string, number>;
}

// Custom Node component to render target/source handles on all four sides (especially left/right)
const CustomNode = ({ data }: any) => {
  return (
    <div
      style={{
        background: '#fff',
        color: '#1e293b',
        border: '2px solid #64748b',
        borderRadius: 8,
        fontWeight: 700,
        fontSize: 14,
        width: 80,
        height: 40,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      {/* Target Handles */}
      <Handle type="target" position={Position.Top} id="top" style={{ opacity: 0 }} />
      <Handle type="target" position={Position.Left} id="left" style={{ opacity: 0 }} />
      <Handle type="target" position={Position.Bottom} id="bottom" style={{ opacity: 0 }} />
      <Handle type="target" position={Position.Right} id="right" style={{ opacity: 0 }} />
      
      {/* Source Handles */}
      <Handle type="source" position={Position.Top} id="top-src" style={{ opacity: 0 }} />
      <Handle type="source" position={Position.Left} id="left-src" style={{ opacity: 0 }} />
      <Handle type="source" position={Position.Bottom} id="bottom-src" style={{ opacity: 0 }} />
      <Handle type="source" position={Position.Right} id="right-src" style={{ opacity: 0 }} />
      
      {data.label}
    </div>
  );
};

const nodeTypes = {
  customNode: CustomNode,
};

// Custom Edge component to render label and colors with perpendicular offset for parallel edges
const CustomEdge = ({
  sourceX,
  sourceY,
  targetX,
  targetY,
  data,
  markerEnd,
}: EdgeProps) => {
  // Calculate direction vector
  const dx = targetX - sourceX;
  const dy = targetY - sourceY;
  const len = Math.sqrt(dx * dx + dy * dy);

  // Offset anti-parallel lines perpendicular to their direction (by e.g. 8 pixels)
  // to prevent overlapping lines and labels.
  const offset = 8;
  const px = len > 0 ? (-dy / len) * offset : 0;
  const py = len > 0 ? (dx / len) * offset : 0;

  const shiftedSourceX = sourceX + px;
  const shiftedSourceY = sourceY + py;
  const shiftedTargetX = targetX + px;
  const shiftedTargetY = targetY + py;

  const [edgePath, labelX, labelY] = getStraightPath({
    sourceX: shiftedSourceX,
    sourceY: shiftedSourceY,
    targetX: shiftedTargetX,
    targetY: shiftedTargetY,
  });

  const edgeData = data as any;
  const congestion = edgeData.congestion;
  const threshold = edgeData.threshold;
  const isReference = edgeData.isReference;

  let color = 'gray'; // Reference
  if (!isReference) {
    if (congestion > threshold) {
      color = 'red';
    } else if (congestion >= threshold * 0.9) {
      color = 'orange';
    } else {
      color = 'green';
    }
  }

  return (
    <>
      <BaseEdge path={edgePath} markerEnd={markerEnd} style={{ stroke: color, strokeWidth: 3 }} />
      <EdgeLabelRenderer>
        <div
          style={{
            position: 'absolute',
            transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
            background: 'white',
            padding: 4,
            borderRadius: 4,
            fontSize: 12,
            fontWeight: 700,
            border: `2px solid ${color}`,
            pointerEvents: 'all',
          }}
          className="nodrag nopan"
        >
          {Math.round(congestion)} / {Math.round(threshold)}
        </div>
      </EdgeLabelRenderer>
    </>
  );
};

const edgeTypes = {
  custom: CustomEdge,
};

// Map each edge to its ideal start/end handles for optimal straight-line visualization
const EDGE_HANDLE_MAP: Record<string, { sourceHandle: string; targetHandle: string }> = {
  "A→B": { sourceHandle: "right-src", targetHandle: "left" },
  "B→A": { sourceHandle: "left-src", targetHandle: "right" },
  "A→C": { sourceHandle: "bottom-src", targetHandle: "top" },
  "C→A": { sourceHandle: "top-src", targetHandle: "bottom" },
  "A→D": { sourceHandle: "right-src", targetHandle: "left" },
  "D→A": { sourceHandle: "left-src", targetHandle: "right" },
  "C→D": { sourceHandle: "right-src", targetHandle: "left" },
  "D→C": { sourceHandle: "left-src", targetHandle: "right" },
  "B→D": { sourceHandle: "bottom-src", targetHandle: "top" },
  "D→B": { sourceHandle: "top-src", targetHandle: "bottom" },
  "D→E": { sourceHandle: "right-src", targetHandle: "left" },
  "E→D": { sourceHandle: "left-src", targetHandle: "right" },
  "E→B": { sourceHandle: "top-src", targetHandle: "right" },
  "B→E": { sourceHandle: "right-src", targetHandle: "top" },
  "B→C": { sourceHandle: "left-src", targetHandle: "top" },
  "C→B": { sourceHandle: "right-src", targetHandle: "bottom" },
};

export const NetworkGraph: React.FC<NetworkGraphProps> = ({ nodesData, edgesData, congestionData, thresholds }) => {
  // Statically define positions for nodes
  const rfNodes = useMemo(() => {
    const defaultPositions: Record<string, { x: number, y: number }> = {
      'A': { x: 50, y: 50 },
      'B': { x: 350, y: 50 },
      'C': { x: 50, y: 250 },
      'D': { x: 350, y: 250 },
      'E': { x: 650, y: 250 },
    };

    const activeNodes = nodesData || [];
    if (activeNodes.length === 0) {
      // Fallback: extract unique node IDs from edgesData
      const uniqueNodeIds = new Set<string>();
      edgesData.forEach((e) => {
        uniqueNodeIds.add(e.source);
        uniqueNodeIds.add(e.target);
      });
      return Array.from(uniqueNodeIds).map((id) => ({
        id,
        type: 'customNode',
        position: defaultPositions[id] || { x: 100, y: 100 },
        data: { label: `Node ${id}` },
      }));
    }

    return activeNodes.map((node) => ({
      id: node.id,
      type: 'customNode',
      position: defaultPositions[node.id] || { x: 100, y: 100 },
      data: { label: `Node ${node.id}` },
    }));
  }, [nodesData, edgesData]);

  const rfEdges: ReactFlowEdge[] = useMemo(() => {
    return edgesData.map((edge) => {
      const congestion = congestionData[edge.id] || 0;
      const threshold = thresholds[edge.id] || 0;
      const handles = EDGE_HANDLE_MAP[edge.id] || { sourceHandle: "bottom-src", targetHandle: "top" };

      return {
        id: edge.id,
        source: edge.source,
        target: edge.target,
        sourceHandle: handles.sourceHandle,
        targetHandle: handles.targetHandle,
        type: 'custom',
        data: {
          congestion,
          threshold,
          isReference: edge.is_reference,
        },
        markerEnd: {
          type: MarkerType.ArrowClosed,
        },
      };
    });
  }, [edgesData, congestionData, thresholds]);

  return (
    <div style={{ width: '100%', height: '400px', border: '1px solid #ccc', borderRadius: '8px' }}>
      <ReactFlow nodes={rfNodes} edges={rfEdges} edgeTypes={edgeTypes} nodeTypes={nodeTypes} fitView>
        <Background />
        <Controls />
      </ReactFlow>
    </div>
  );
};
