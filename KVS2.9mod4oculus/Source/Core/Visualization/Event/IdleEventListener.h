/*****************************************************************************/
/**
 *  @file   IdleEventListener.h
 *  @author Naohisa Sakamoto
 */
/*----------------------------------------------------------------------------
 *
 *  Copyright (c) Visualization Laboratory, Kyoto University.
 *  All rights reserved.
 *  See http://www.viz.media.kyoto-u.ac.jp/kvs/copyright/ for details.
 *
 *  $Id: IdleEventListener.h 1325 2012-10-04 10:34:52Z naohisa.sakamoto@gmail.com $
 */
/*****************************************************************************/
#ifndef KVS__IDLE_EVENT_LISTENER_H_INCLUDE
#define KVS__IDLE_EVENT_LISTENER_H_INCLUDE

#include <kvs/EventListener>
#include <kvs/EventBase>


namespace kvs
{

/*===========================================================================*/
/**
 *  @brief  IdleEventListener class.
 */
/*===========================================================================*/
class IdleEventListener : public kvs::EventListener
{
public:

    IdleEventListener();
    virtual ~IdleEventListener();

    virtual void update() = 0;

private:

    void onEvent( kvs::EventBase* event = 0 );
};

} // end of namespace kvs

#endif // KVS__IDLE_EVENT_LISTENER_H_INCLUDE
