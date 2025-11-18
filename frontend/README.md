# Digital Twin Factory - Frontend

React + TypeScript dashboard for real-time factory monitoring.

## Features

- **Real-time Monitoring**: WebSocket-based live updates
- **3D Digital Twin Visualization**: Interactive 3D factory scene with real-time updates
- **2D Dashboard**: Traditional dashboard with charts and metrics
- **Predictive Maintenance**: ML-based maintenance recommendations
- **Production Scheduling**: Interactive schedule timeline
- **Responsive Design**: Tailwind CSS with mobile support
- **Split View Mode**: Combine 3D and 2D views simultaneously

## Tech Stack

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **Three.js** - 3D graphics engine
- **React Three Fiber** - React renderer for Three.js
- **React Three Drei** - Useful helpers for R3F
- **Axios** - HTTP client
- **Recharts** - Data visualization
- **Lucide React** - Icons
- **React Router** - Navigation

## Getting Started

### Install Dependencies

```bash
npm install
```

### Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

### Build for Production

```bash
npm run build
```

### Preview Production Build

```bash
npm run preview
```

## Project Structure

```
frontend/
├── src/
│   ├── components/         # React components
│   │   ├── Dashboard/     # Dashboard views
│   │   ├── DigitalTwin/   # Digital twin 2D components
│   │   ├── DigitalTwin3D/ # 3D visualization components
│   │   ├── Predictive/    # Predictive maintenance
│   │   ├── Scheduling/    # Scheduling components
│   │   └── common/        # Reusable UI components
│   ├── services/          # API clients and WebSocket
│   ├── types/             # TypeScript type definitions
│   ├── utils/             # Helper functions
│   │   ├── three/         # Three.js utilities
│   │   └── helpers.ts     # General helpers
│   ├── App.tsx            # Main app component
│   └── main.tsx           # Entry point
├── public/                # Static assets
└── package.json           # Dependencies
```

## API Integration

The frontend connects to the FastAPI backend at `http://localhost:8000/api/v1`.

API endpoints:
- `/predict` - Image defect detection
- `/digital-twin/factory/:id` - Factory state
- `/predictive/maintenance/:id` - Maintenance recommendations
- `/scheduling/current` - Current schedule

WebSocket endpoint:
- `ws://localhost:8000/ws/stream` - Real-time updates

## Environment Variables

Create a `.env` file:

```env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

## Development

### Type Checking

```bash
npm run type-check
```

### Linting

```bash
npm run lint
```

## Component Examples

### Using API Client

```typescript
import { apiClient } from '@/services/api';

const stats = await apiClient.getDashboardStats();
```

### Using WebSocket

```typescript
import { wsService } from '@/services/websocket';

wsService.on('machine_update', (data) => {
  console.log('Machine updated:', data);
});
```

## Deployment

Build the production bundle:

```bash
npm run build
```

Serve the `dist/` directory with any static file server.

## License

MIT
